"""The extension API package: one versioned door for pack authors.

Packs import ``dinkster_api.v1`` and nothing else (DESIGN 3.6). Internal
packages (`dinkster_schema`, `dinkster_values`, ...) are reachable in Python,
but only what `v1` exports is covered by the compat test suite and the
additive-only promise. Import the version you target explicitly::

    from dinkster_api.v1 import Node, NodeSchema, InputSpec, OutputSpec, TypeExpr
"""
