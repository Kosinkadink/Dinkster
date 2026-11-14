"""remotes.toml parsing: strict, loud, and shaped exactly like the module
docstring's grammar - [worker.NAME] tables with endpoint, token_file,
optional nodes allowlist, optional trust_reserved, optional [worker.NAME.
memory] budgets in the remote's OWN device namespace (the host adds the
@name qualifier later, so '@' anywhere in the config is misconfiguration).
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from dinkster.remotes import RemotesError, RemoteSpec, load_remotes, parse_remotes


def parse(text: str) -> tuple[RemoteSpec, ...]:
    return parse_remotes(tomllib.loads(text), "remotes.toml")


def test_happy_path_parses_every_field() -> None:
    specs = parse(
        """
        [worker.upscale-box]
        endpoint = "192.168.1.53:5151"
        token_file = "/etc/dinkster/upscale-box.token"
        nodes = ["esrgan.upscale", "esrgan.face"]
        trust_reserved = true

        [worker.upscale-box.memory]
        ram = "24G"
        "vram:cuda:0" = "20G"
        """
    )
    assert len(specs) == 1
    spec = specs[0]
    assert spec.name == "upscale-box"
    assert spec.host == "192.168.1.53"
    assert spec.port == 5151
    assert spec.token_file == Path("/etc/dinkster/upscale-box.token")
    assert spec.nodes == ("esrgan.upscale", "esrgan.face")
    assert spec.trust_reserved is True
    assert spec.memory_budgets == {"ram": 24 * 1024**3, "vram:cuda:0": 20 * 1024**3}


@pytest.mark.parametrize("value", ['""', "5", "true", '["a.pem"]'])
def test_tls_ca_file_must_be_a_non_empty_path_string(value: str) -> None:
    with pytest.raises(RemotesError, match="'tls_ca_file' must be a non-empty path string"):
        parse(
            f"""
            [worker.box]
            endpoint = "10.0.0.2:5151"
            token_file = "/tmp/token"
            tls_ca_file = {value}
            """
        )


def test_unknown_top_level_key_refused() -> None:
    with pytest.raises(RemotesError, match="unknown top-level keys"):
        parse('[workers.box]\nendpoint = "h:1"\ntoken_file = "/t"\n')


def test_unknown_worker_key_refused() -> None:
    with pytest.raises(RemotesError, match="unknown keys"):
        parse('[worker.box]\nendpoint = "h:1"\ntoken_file = "/t"\nretries = 3\n')


def test_integer_memory_size_is_bytes() -> None:
    specs = parse(
        '[worker.box]\nendpoint = "h:1"\ntoken_file = "/t"\n[worker.box.memory]\nram = 1024\n'
    )
    assert specs[0].memory_budgets == {"ram": 1024}


def test_load_remotes_missing_file_means_no_remotes(tmp_path: Path) -> None:
    assert load_remotes(tmp_path / "absent.toml") == ()


def test_load_remotes_reads_file(tmp_path: Path) -> None:
    path = tmp_path / "remotes.toml"
    path.write_text(
        '[worker.box]\nendpoint = "10.0.0.2:5151"\ntoken_file = "/tmp/token"\n',
        encoding="utf-8",
    )
    (spec,) = load_remotes(path)
    assert (spec.name, spec.host, spec.port) == ("box", "10.0.0.2", 5151)
