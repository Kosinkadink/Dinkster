#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if ! command -v uv >/dev/null; then
    uv_dir="$HOME/.local/bin"
    if [ ! -x "$uv_dir/uv" ]; then
        echo 'Installing uv in your user directory (no sudo or shell-profile changes)...'
        if command -v curl >/dev/null; then
            curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$uv_dir" UV_NO_MODIFY_PATH=1 sh || {
                echo 'error: uv installation failed; check Internet access to https://astral.sh/uv/install.sh' >&2
                exit 1
            }
        elif command -v wget >/dev/null; then
            wget -qO- https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$uv_dir" UV_NO_MODIFY_PATH=1 sh || {
                echo 'error: uv installation failed; check Internet access to https://astral.sh/uv/install.sh' >&2
                exit 1
            }
        else
            echo 'error: curl or wget is required to install uv' >&2
            exit 1
        fi
    fi
    export PATH="$uv_dir:$PATH"
fi
exec uv run --no-project --python 3.12 scripts/run.py "$@"
