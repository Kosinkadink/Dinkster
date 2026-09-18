"""Generate SaveText goldens from ComfyUI e20d433a.

Usage from the Dinkster repository root with an interpreter containing
ComfyUI's declared dependencies:

    COMFYUI_ROOT=/path/to/ComfyUI \
      /path/to/python tools/gen_text_io_goldens.py

The ComfyUI checkout must be clean and pinned to the commit below. Run the
generator twice and compare the printed sha256 before committing the fixture.

Each case stores the exact text fed to the pinned SaveTextNode (base64) plus
the saved file's name and bytes, so the Dinkster-side tests replay identical
inputs and compare output bytes directly.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

BASELINE = "e20d433a4966dcc88fa5abbae6ace824cb78b263"
REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "tests" / "goldens" / "text_io_e20d433a.json"


def _git(comfy_root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(comfy_root), *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


CASES = [
    ("txt_plain_ascii", "txt", "plain body with no trailing newline"),
    ("txt_multiline_unicode", "txt", "first line\nsecond line with e-acute \u00e9\n"),
    ("csv_rows", "csv", "a,b,c\n1,2,3\n"),
    ("md_heading", "md", "# Title\n\nBody paragraph.\n"),
    ("json_valid_pretty", "json", '{"b":2,"a":[1,2],"s":"caf\u00e9"}'),
    ("json_invalid_raw", "json", "not json {"),
    ("json_scalar", "json", "42"),
]


def build_goldens(comfy_root: Path) -> dict[str, object]:
    if _git(comfy_root, "rev-parse", "HEAD") != BASELINE:
        raise RuntimeError(f"ComfyUI must be pinned to {BASELINE}")
    if _git(comfy_root, "status", "--porcelain"):
        raise RuntimeError("ComfyUI checkout must be clean")
    sys.path.insert(0, str(comfy_root))

    import folder_paths  # pyright: ignore[reportMissingImports]
    from comfy_extras import nodes_text  # pyright: ignore[reportMissingImports]

    cases: dict[str, object] = {}
    with tempfile.TemporaryDirectory() as scratch:
        folder_paths.set_output_directory(scratch)
        for name, format_name, text in CASES:
            result = nodes_text.SaveTextNode.execute(text, f"golden/{name}", format_name)
            saved = result.ui["files"][0]
            path = Path(scratch) / saved.subfolder / saved.filename
            cases[name] = {
                "format": format_name,
                "textB64": _b64(text.encode("utf-8")),
                "filename": saved.filename,
                "passthrough": result.result[0] == text,
                "fileB64": _b64(path.read_bytes()),
            }

        # Counter behavior: a second save with an already-used prefix.
        nodes_text.SaveTextNode.execute("first", "golden/counter", "txt")
        second = nodes_text.SaveTextNode.execute("second", "golden/counter", "txt")
        cases["counter_second_save"] = {
            "filename": second.ui["files"][0].filename,
        }

    return {
        "revision": BASELINE,
        "generator": "tools/gen_text_io_goldens.py",
        "cases": cases,
    }


def main() -> None:
    configured = os.environ.get("COMFYUI_ROOT")
    comfy_root = Path(configured) if configured else REPO.parent / "ComfyUI"
    content = (json.dumps(build_goldens(comfy_root.resolve()), indent=2) + "\n").encode()
    OUT.write_bytes(content)
    print(f"wrote {OUT} sha256={hashlib.sha256(content).hexdigest()}")


if __name__ == "__main__":
    main()
