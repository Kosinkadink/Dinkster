from collections.abc import Iterator
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest


@pytest.fixture
def unix_socket_dir() -> Iterator[Path]:
    # macOS pytest tmp_path names can exceed the Unix socket address limit.
    with TemporaryDirectory(prefix="dinkster-sock-", dir="/tmp") as directory:
        yield Path(directory)
