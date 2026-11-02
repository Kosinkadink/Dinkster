"""Minimal installed requirements reader for the lifecycle fixture engine."""

import hashlib
import json
from types import SimpleNamespace


def load_project_requirements(root):
    data = (root / "project.json").read_bytes()
    requirements = json.loads(data).get("additions", {})
    return SimpleNamespace(to_wire=lambda: requirements), [hashlib.sha256(data).hexdigest()]
