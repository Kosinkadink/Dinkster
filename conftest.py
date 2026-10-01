import os
from collections.abc import Iterator
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from tools.pytest_file_shard import ALL_FILE_SHARDS_MARKER


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        f"{ALL_FILE_SHARDS_MARKER}: run this test on every file shard",
    )


@pytest.fixture
def model_root() -> Path:
    return Path(
        os.environ.get(
            "DINKSTER_PARITY_ARTIFACT_ROOT", str(Path.home() / "ComfyUI-Shared" / "models")
        )
    ).expanduser()


@pytest.fixture
def unix_socket_dir() -> Iterator[Path]:
    # macOS pytest tmp_path names can exceed the Unix socket address limit.
    with TemporaryDirectory(prefix="dinkster-sock-", dir="/tmp") as directory:
        yield Path(directory)
