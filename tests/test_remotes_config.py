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


def test_minimal_entry_defaults() -> None:
    specs = parse(
        """
        [worker.box]
        endpoint = "10.0.0.2:5151"
        token_file = "/tmp/token"
        """
    )
    spec = specs[0]
    assert spec.tls_ca_file is None
    assert spec.nodes is None
    assert spec.trust_reserved is False
    assert spec.memory_budgets == {}


def test_tls_ca_file_parses_to_path() -> None:
    specs = parse(
        """
        [worker.box]
        endpoint = "10.0.0.2:5151"
        token_file = "/tmp/token"
        tls_ca_file = "/etc/dinkster/box.pem"
        """
    )
    assert specs[0].tls_ca_file == Path("/etc/dinkster/box.pem")


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


def test_multiple_workers_preserve_order() -> None:
    specs = parse(
        """
        [worker.a]
        endpoint = "h1:1"
        token_file = "/t1"

        [worker.b]
        endpoint = "h2:2"
        token_file = "/t2"
        """
    )
    assert [spec.name for spec in specs] == ["a", "b"]


def test_unknown_top_level_key_refused() -> None:
    with pytest.raises(RemotesError, match="unknown top-level keys"):
        parse('[workers.box]\nendpoint = "h:1"\ntoken_file = "/t"\n')


def test_unknown_worker_key_refused() -> None:
    with pytest.raises(RemotesError, match="unknown keys"):
        parse('[worker.box]\nendpoint = "h:1"\ntoken_file = "/t"\nretries = 3\n')


@pytest.mark.parametrize("endpoint", ["", "hostonly", ":5151", "h:notaport", "h:0", "h:70000"])
def test_malformed_endpoint_refused(endpoint: str) -> None:
    with pytest.raises(RemotesError, match="endpoint"):
        parse(f'[worker.box]\nendpoint = "{endpoint}"\ntoken_file = "/t"\n')


def test_missing_endpoint_refused() -> None:
    with pytest.raises(RemotesError, match="endpoint"):
        parse('[worker.box]\ntoken_file = "/t"\n')


def test_missing_token_file_refused() -> None:
    with pytest.raises(RemotesError, match="token_file"):
        parse('[worker.box]\nendpoint = "h:1"\n')


def test_empty_token_file_refused() -> None:
    with pytest.raises(RemotesError, match="token_file"):
        parse('[worker.box]\nendpoint = "h:1"\ntoken_file = ""\n')


def test_at_sign_in_worker_name_refused() -> None:
    with pytest.raises(RemotesError, match="'@'"):
        parse('[worker."box@lan"]\nendpoint = "h:1"\ntoken_file = "/t"\n')


def test_whitespace_in_worker_name_refused() -> None:
    with pytest.raises(RemotesError, match="whitespace"):
        parse('[worker."two words"]\nendpoint = "h:1"\ntoken_file = "/t"\n')


@pytest.mark.parametrize("name", ("local", "LOCAL"))
def test_local_worker_name_is_reserved(name: str) -> None:
    with pytest.raises(RemotesError, match="worker name 'local' is reserved"):
        parse(f'[worker.{name}]\nendpoint = "h:1"\ntoken_file = "/t"\n')


def test_at_sign_in_memory_device_refused() -> None:
    with pytest.raises(RemotesError, match="unqualified"):
        parse(
            "[worker.box]\n"
            'endpoint = "h:1"\n'
            'token_file = "/t"\n'
            "[worker.box.memory]\n"
            '"ram@box" = "1G"\n'
        )


@pytest.mark.parametrize("size", ['"garbage"', "-1", "true"])
def test_bad_memory_size_refused(size: str) -> None:
    with pytest.raises(RemotesError, match="memory budget"):
        parse(
            "[worker.box]\n"
            'endpoint = "h:1"\n'
            'token_file = "/t"\n'
            "[worker.box.memory]\n"
            f"ram = {size}\n"
        )


def test_integer_memory_size_is_bytes() -> None:
    specs = parse(
        '[worker.box]\nendpoint = "h:1"\ntoken_file = "/t"\n[worker.box.memory]\nram = 1024\n'
    )
    assert specs[0].memory_budgets == {"ram": 1024}


def test_nodes_must_be_nonempty_strings() -> None:
    with pytest.raises(RemotesError, match="nodes"):
        parse('[worker.box]\nendpoint = "h:1"\ntoken_file = "/t"\nnodes = ["ok", ""]\n')
    with pytest.raises(RemotesError, match="nodes"):
        parse('[worker.box]\nendpoint = "h:1"\ntoken_file = "/t"\nnodes = "notalist"\n')


def test_load_remotes_missing_file_means_no_remotes(tmp_path: Path) -> None:
    assert load_remotes(tmp_path / "absent.toml") == ()


def test_load_remotes_invalid_toml_fails_loudly(tmp_path: Path) -> None:
    path = tmp_path / "remotes.toml"
    path.write_text("[worker.box\n", encoding="utf-8")
    with pytest.raises(RemotesError, match="invalid TOML"):
        load_remotes(path)


def test_load_remotes_reads_file(tmp_path: Path) -> None:
    path = tmp_path / "remotes.toml"
    path.write_text(
        '[worker.box]\nendpoint = "10.0.0.2:5151"\ntoken_file = "/tmp/token"\n',
        encoding="utf-8",
    )
    (spec,) = load_remotes(path)
    assert (spec.name, spec.host, spec.port) == ("box", "10.0.0.2", 5151)
