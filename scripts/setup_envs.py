"""Install the machine's published base and locked editable contributor workspace."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def command(*args: str, root: Path = ROOT, env: dict[str, str] | None = None) -> None:
    environment = os.environ.copy()
    for name in ("UV_PROJECT", "UV_PROJECT_ENVIRONMENT"):
        environment.pop(name, None)
    environment.update(env or {})
    # Windows bare-name process lookup does not honor PATH's .cmd shims.
    executable = shutil.which(args[0], path=environment.get("PATH")) or args[0]
    subprocess.run((executable, *args[1:]), cwd=root, check=True, env=environment)


def bootstrap(root: Path) -> Path:
    """The lock's supervisor closure contains no torch or accelerator dependencies."""
    environment = root / ".run/bootstrap"
    command(
        "uv",
        "sync",
        "--locked",
        "--package",
        "dinkster-supervisor",
        "--no-dev",
        root=root,
        env={"UV_PROJECT_ENVIRONMENT": str(environment), "UV_PROJECT": str(root)},
    )
    return environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def select_base(bases: list[dict], variant: str | None = None) -> dict:
    host = {"Windows": "windows", "Linux": "linux", "Darwin": "macos"}[platform.system()]
    arch = {"AMD64": "x86_64", "aarch64": "arm64"}.get(platform.machine(), platform.machine())
    if host == "macos" and arch == "x86_64":
        raise RuntimeError("Intel macOS is unsupported; mac-arm64 requires Apple Silicon.")
    compatible = [base for base in bases if base["platform"] == {"os": host, "arch": arch}]
    mask = os.environ.get("CUDA_VISIBLE_DEVICES", "unset").strip()
    devices = []
    if mask not in ("", "-1") and any(base["nvidia"] for base in compatible):
        try:
            text = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=driver_version,compute_cap", "--format=csv,noheader"],
                text=True,
                timeout=15,
                stderr=subprocess.DEVNULL,
            )
            devices = [
                tuple(part.strip() for part in line.split(",")) for line in text.splitlines()
            ]
        except (OSError, subprocess.SubprocessError):
            pass

    def version(value: str) -> tuple[int, ...]:
        return tuple(int(part) for part in value.split("."))

    def usable(base: dict) -> bool:
        bounds = base["nvidia"]
        if bounds is None:
            return True
        return bool(devices) and all(
            version(driver) >= version(bounds["minimumDriver"])
            and version(capability) >= version(bounds["minimumComputeCapability"])
            and (
                bounds["maximumExclusiveComputeCapability"] is None
                or version(capability) < version(bounds["maximumExclusiveComputeCapability"])
            )
            for driver, capability in devices
        )

    candidates = [
        base
        for base in compatible
        if usable(base) and (variant is None or base["variant"] == variant)
    ]
    if not candidates:
        raise RuntimeError(
            "No published base supports this machine/selection; see Dinkster-Environments."
        )
    return max(
        candidates,
        key=lambda base: (
            base["nvidia"] is not None,
            version(base["nvidia"]["minimumDriver"]) if base["nvidia"] else (),
        ),
    )


def setup(
    root: Path,
    *,
    variant: str | None = None,
    mirror: str = "https://releases.dinkster.org",
    force: bool = False,
) -> Path:
    from dinkster_supervisor.environment_artifacts import acquire_base, validate_manifest
    from dinkster_supervisor.environment_constraints import effective_constraints
    from dinkster_supervisor.environment_downloads import read_metadata
    from dinkster_supervisor.environment_state import writer_lock

    selections = json.loads((root / "scripts/base_bundles.json").read_text())
    bases = [
        validate_manifest(
            read_metadata(mirror, f"base/{name}/{identity}/manifest.json"),
            "sha256:" + identity,
            name,
        )
        for name, identity in selections.items()
    ]
    base = select_base(bases, variant)
    print(f"Installing {base['variant']} base {base['id']} (Python {base['python']})", flush=True)
    store = root / ".run/environment-store"
    environment = root / ".venv"
    store.mkdir(parents=True, exist_ok=True)
    with writer_lock(store):
        base_root = acquire_base(
            base, (f"{mirror.rstrip('/')}/base/{base['variant']}/{base['id'][7:]}",), store
        )
        requirements = root / ".run/workspace-requirements.txt"
        command(
            "uv",
            "export",
            "--locked",
            "--all-packages",
            "--no-emit-workspace",
            "--no-annotate",
            "--no-header",
            "--output-file",
            str(requirements),
            root=root,
        )
        requirements.write_text(effective_constraints(requirements.read_text(), base))
        receipt = root / ".run/environment-base.json"
        if environment.exists() and (
            force or not receipt.is_file() or json.loads(receipt.read_text())["id"] != base["id"]
        ):
            shutil.rmtree(environment)
        command(
            "uv",
            "venv",
            "--python",
            str(base_root / base["interpreter"]["executable"]),
            "--allow-existing",
            str(environment),
            root=root,
        )
        python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        command(
            "uv",
            "pip",
            "sync",
            "--python",
            str(python),
            "--require-hashes",
            "--find-links",
            str(base_root / "wheels"),
            str(requirements),
            root=root,
        )
        # The exported lock owns dependencies; editable metadata must not resolve them again.
        from scripts.build_release import workspace_projects

        editables = [
            argument
            for path in workspace_projects(root).values()
            for argument in ("--editable", str(path.parent))
        ]
        command("uv", "pip", "install", "--python", str(python), "--no-deps", *editables, root=root)
        evidence = Path(
            os.environ.get("DINKSTER_EVIDENCE_ROOT", root.parent / ".dinkster-evidence-source")
        )
        acceptance = evidence / "packages/dinkster-acceptance"
        if acceptance.is_dir():
            command(
                "uv",
                "pip",
                "install",
                "--python",
                str(python),
                "--no-deps",
                "--no-sources",
                "--editable",
                str(acceptance),
                root=root,
            )
        receipt.write_text(json.dumps(base, indent=2) + "\n")
    return python


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant")
    parser.add_argument("--mirror", default="https://releases.dinkster.org")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if not shutil.which("uv"):
        parser.error("uv is required: https://docs.astral.sh/uv/")
    if not os.environ.get("DINKSTER_SETUP_BOOTSTRAPPED"):
        python = bootstrap(ROOT)
        command(
            str(python),
            str(Path(__file__).resolve()),
            *sys.argv[1:],
            env={"DINKSTER_SETUP_BOOTSTRAPPED": "1"},
        )
        return 0
    sys.path.insert(0, str(ROOT))
    python = setup(ROOT, variant=args.variant, mirror=args.mirror, force=args.force)
    print(f"Ready: {python} (use --no-sync with uv run)")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, ValueError, subprocess.CalledProcessError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1) from None
