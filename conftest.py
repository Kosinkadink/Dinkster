import importlib.util
import os
from collections.abc import Iterator
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from tools.pytest_file_shard import ALL_FILE_SHARDS_MARKER


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--p2p", action="store_true", help="run optional P2P tests (requires p2p extra)"
    )


def pytest_ignore_collect(collection_path: Path, config: pytest.Config) -> bool | None:
    name = collection_path.name
    if name.startswith("test_") and ("p2p" in name or name.startswith("test_seed_service")):
        return not config.getoption("--p2p")
    return None


def pytest_configure(config: pytest.Config) -> None:
    if config.getoption("--p2p") and importlib.util.find_spec("dinkster_p2p") is None:
        raise pytest.UsageError("--p2p requires installation with the p2p extra")
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
