"""Survey legacy ComfyUI packs through the quarantine loader (DESIGN 3.8).

For each pack directory given, spawn one fresh child on the ComfyUI
install's interpreter, run dinkster_compat_comfy.legacy.load_legacy_packs()
against just that pack, and collect its LegacyPackReport JSON. One process
per pack on purpose: a pack that crashes the interpreter or poisons
sys.modules must not contaminate the next pack's report.

The point of the survey (per project direction 2026-07) is less "does it
load" than "what did the pack have to hook or patch" - the reports feed
the hack census that shapes Dinkster's extension API. Compatibility numbers
are the side benefit.

Usage:

    python tools/legacy_survey.py --comfyui-root /home/kosin/ComfyUI \
        --out /tmp/dinkster-legacy-survey /tmp/dinkster-legacy-packs/*

Stdlib only; runnable by any Python >= 3.12. The child needs Dinkster's
pure-stdlib packages on PYTHONPATH; this script wires that from its own
repo location.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

CHILD_CODE = "import dinkster_compat_comfy.legacy as l; l.load_legacy_packs()"


def dinkster_pythonpath(extra: Sequence[Path] = ()) -> str:
    """Child PYTHONPATH: optional dependency overlays first, Dinkster last.

    An overlay is a ``pip install --target`` directory holding deps the
    packs need but the ComfyUI venv lacks. Build it with the venv's own
    pip and ``--no-deps`` (plus explicitly-listed transitive deps): a full
    resolve drags in a second torch/numpy that would shadow the venv's and
    break every pack at once."""
    parts = [str(p) for p in extra]
    parts.extend(sorted(str(p) for p in (REPO_ROOT / "packages").glob("*/src")))
    return os.pathsep.join(parts)


def comfy_python(root: Path) -> str:
    explicit = os.environ.get("DINKSTER_COMFYUI_PYTHON", "")
    if explicit:
        return explicit
    venv = root / "venv" / "bin" / "python"
    if venv.exists():
        return str(venv)
    return sys.executable


def survey_pack(
    pack: Path,
    *,
    python: str,
    comfyui_root: Path,
    out_dir: Path,
    timeout: float,
    overlays: Sequence[Path] = (),
) -> dict[str, object]:
    report_path = out_dir / f"{pack.name}.json"
    report_path.unlink(missing_ok=True)
    env = {
        **os.environ,
        "DINKSTER_COMFYUI_ROOT": str(comfyui_root),
        "DINKSTER_LEGACY_PACKS": str(pack),
        "DINKSTER_LEGACY_REPORT": str(report_path),
        "PYTHONPATH": dinkster_pythonpath(overlays),
    }
    started = time.monotonic()
    try:
        proc = subprocess.run(
            [python, "-c", CHILD_CODE],
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        exit_code: int | None = proc.returncode
        stderr_tail = proc.stderr[-4000:]
    except subprocess.TimeoutExpired as exc:
        exit_code = None
        stderr_text = (
            exc.stderr.decode("utf-8", "replace")
            if isinstance(exc.stderr, bytes)
            else (exc.stderr or "")
        )
        stderr_tail = stderr_text[-4000:]
    elapsed = round(time.monotonic() - started, 1)

    record: dict[str, object] = {
        "pack": pack.name,
        "path": str(pack),
        "elapsed_s": elapsed,
        "exit_code": exit_code,
        "stderr_tail": stderr_tail,
    }
    if report_path.is_file():
        reports = json.loads(report_path.read_text(encoding="utf-8"))
        record["report"] = reports[0] if reports else None
        record["status"] = reports[0]["status"] if reports else "empty-report"
    elif exit_code is None:
        record["status"] = "timeout"
    else:
        # The loader always writes a report when it runs at all; no file
        # means the child died before/inside bootstrap.
        record["status"] = "child-crashed"
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("packs", nargs="+", type=Path, help="pack directories")
    parser.add_argument("--comfyui-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="output directory")
    parser.add_argument("--timeout", type=float, default=420.0)
    parser.add_argument(
        "--overlay",
        type=Path,
        action="append",
        default=[],
        help="pip --target overlay dir(s) prepended to the child PYTHONPATH",
    )
    args = parser.parse_args()

    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    python = comfy_python(args.comfyui_root)

    records: list[dict[str, object]] = []
    for pack in args.packs:
        if not pack.is_dir():
            print(f"skip (not a directory): {pack}", file=sys.stderr)
            continue
        print(f"surveying {pack.name} ...", flush=True)
        record = survey_pack(
            pack,
            python=python,
            comfyui_root=args.comfyui_root,
            out_dir=out_dir,
            timeout=args.timeout,
            overlays=args.overlay,
        )
        records.append(record)
        report = record.get("report")
        detail = ""
        if isinstance(report, dict):
            detail = (
                f" nodes={report['nodes_translated']}"
                f" skipped={len(report['nodes_skipped'])}"
                f" routes={report['server_routes_added']}"
                f" hooks={','.join(report['hook_imports']) or '-'}"
            )
            if report.get("error"):
                detail += f" error={report['error'][:120]}"
        print(f"  {record['status']} ({record['elapsed_s']}s){detail}", flush=True)

    summary_path = out_dir / "survey.json"
    summary_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"\nwrote {summary_path}")

    by_status: dict[str, int] = {}
    for record in records:
        status = str(record["status"])
        by_status[status] = by_status.get(status, 0) + 1
    for status, count in sorted(by_status.items()):
        print(f"  {status}: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
