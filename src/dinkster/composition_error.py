"""Composition refusals shared by serving and installation commands."""


class CompositionError(Exception):
    """Pack composition is misconfigured: reserved/duplicate pack names or
    node-type collisions. Raised at startup, never mid-workflow."""
