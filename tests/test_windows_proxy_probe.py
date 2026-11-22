"""Diagnostic only: separate config lookup from HTTPS loopback routing."""

import pygit2
from dinkster_supervisor.environment_state import EnvironmentError
from dinkster_supervisor.git_sources import _isolated_config, fetched_git_source, git_refs


def test_observe_https_proxy_routes(git_server):
    print("libgit2 version:", pygit2.LIBGIT2_VERSION, flush=True)
    git_server["config"].write_text(f"[http]\n proxy = {git_server['proxy']}\n")
    configured = pygit2.Config.get_global_config()["http.proxy"]
    print("Global config matches generated proxy:", configured == git_server["proxy"], flush=True)
    with _isolated_config() as selected:
        print("Product selects generated proxy:", selected == git_server["proxy"], flush=True)
    for host in ("localhost", "source.example.invalid"):
        url = git_server["url"].replace("localhost", host)
        before = len(git_server["tunnels"])
        try:
            with fetched_git_source(url) as (folder, origin):
                assert (folder / "pack/node.py").read_text() == "result = 17\n"
                assert origin["commit"] == str(git_server["commits"]["main"])
            refs = git_refs(url)
            assert refs["refs/heads/main"] == str(git_server["commits"]["main"])
        except EnvironmentError as error:
            print("Host", host, "product refusal:", str(error), flush=True)
        else:
            print(
                "Host",
                host,
                "verified content and refs; new tunnels:",
                len(git_server["tunnels"]) - before,
                flush=True,
            )
    print("Diagnostic finished; original proxy assertion was not modified", flush=True)
