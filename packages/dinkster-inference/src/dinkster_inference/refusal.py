"""Stable native-planning refusal vocabulary shared by planners and dispatch."""

from enum import StrEnum


class NativeRefusalCategory(StrEnum):
    """Stable native-planning categories for strict dispatch.

    ``native_ineligible`` is reserved for a completely and successfully
    planned composition that the ordinary owner can run but the requested
    native capability cannot. Invalid, ambiguous, incomplete, identity,
    configuration, and unknown cases fail closed as
    ``invalid_input_or_identity``. Reasons are diagnostics only and must
    never be parsed for control flow.
    """

    INVALID_INPUT_OR_IDENTITY = "invalid_input_or_identity"
    NATIVE_INELIGIBLE = "native_ineligible"


def _require_refusal_category(category: object) -> NativeRefusalCategory:
    if not isinstance(category, NativeRefusalCategory):
        raise TypeError("category must be a NativeRefusalCategory")
    return category


class NativeRefusalError(Exception):
    """A typed refusal to construct a native assembly plan.

    ``native_ineligible`` requires a complete valid ordinary-owner plan
    unsupported by the requested native capability. Invalid, ambiguous,
    incomplete, identity, configuration, and unknown cases fail closed as
    ``invalid_input_or_identity``. ``reasons`` are diagnostics only; callers
    branch on ``category`` and never parse reason text.
    """

    def __init__(
        self,
        reasons: tuple[str, ...],
        category: NativeRefusalCategory = NativeRefusalCategory.INVALID_INPUT_OR_IDENTITY,
    ) -> None:
        self.reasons = reasons
        self.category = _require_refusal_category(category)
        super().__init__("; ".join(reasons))


__all__ = ["NativeRefusalCategory", "NativeRefusalError"]
