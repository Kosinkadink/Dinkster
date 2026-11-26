# Native environment tools build and bootstrap

The native tools payload is independent of the selected engine release. Use the
supervisor package's own release-built wheel, not an engine version or a newly
resolved PyPI version. An unpublished local output is not a production download
pin. See [channel signing](channel-signing.md) before publishing channels.

## Build native tools

Run from the repository root with the same native Python patch version,
implementation, architecture and ABI as the standalone interpreter, including
free-threaded status:

```sh
uv run --locked python -m scripts.build_environment_tools \
  --interpreter /absolute/path/cpython-install-only.tar.gz \
  --interpreter-size BYTES --interpreter-sha256 LOWERCASE_SHA256 \
  --supervisor-wheel /absolute/path/dinkster_supervisor-VERSION-py3-none-any.whl \
  --output /absolute/new/tools-output
```

The interpreter input can also be an HTTPS URL. Obtain its complete size and
SHA256 from the selected standalone interpreter's verified producer input, not
a latest alias. The builder selects compatible locked supervisor/tools wheels;
no engine or torch is installed. The supervisor's `environment-tools` dependency
group pins uv. Inspect the offline dependency-closure check before packaging.

Use `scripts/build_tools_release.py` to finalize the three native platform ZIPs
into the tools manifest and checksums. The independent Supervisor tools publication
workflow publishes the immutable tools release and moves the Nightly tools
pointer; an engine release does not supply these ZIPs.

## Acquire and launch

Inspect the selection without `--yes`, then acquire it explicitly:

```sh
dinkster-env --root /absolute/tools-cache tools --channel nightly \
  --mirror https://releases.dinkster.org --yes --json
```

Desktop must select the exact native archive from its pinned tools manifest or
the channel's tools publication, verify its full size/SHA256, and safely extract
it into a new owned directory. Preserve Unix executable mode bits from ZIP
attributes, reject escaping paths and never overwrite an existing installation.
Internal aliases are materialized, so Windows needs no symlink privilege. ZIP is
the bootstrap container, not a replacement for the producer's tar+zstd base
contract. Extraction must not run pip, uv or package resolution.

Invoke `python/bin/python3 -I -m dinkster_supervisor.environment` relative to the
tools directory on Linux/macOS, or `python/python.exe -I -m
dinkster_supervisor.environment` on Windows. Prepend that child's `python/bin`
(Windows `python/Scripts`) to its PATH for bundled uv; do not change global PATH.
Console-script shebangs are not the relocatable interface.

The [lifecycle fixture builder](../tests/fixtures/environments/README.md) supplies
tiny production-format mirror inputs for CLI/Desktop equivalence. It is separate
from this tools builder and must not be used as production release artifacts.

## Channels and compatibility

For a local channel document, use `python -m dinkster_supervisor.environment_channels`
with `--release RELEASE_DIRECTORY`,
repeated `--base-manifest BASE_MANIFEST`, `--channel nightly` (or `stable`),
`--output-root MIRROR_DIRECTORY` and `--minimum-desktop-version X.Y.Z`.
The builder is local and nonpublishing; the release workflow's publication job
performs public payload verification before replacing the channel.

The release workflow's `DINKSTER_MINIMUM_DESKTOP_VERSION` owns the compatibility
floor. Raise it only when engine layout or lifecycle compatibility needs a newer
Desktop, and publish that Desktop before making the channel require it. Artifact
layouts belong to their publishing tools; locations never replace digest checks.

## Reference

- [Native tools builder](../scripts/build_environment_tools.py) and [release finalizer](../scripts/build_tools_release.py)
- [Acquisition](../packages/dinkster-supervisor/src/dinkster_supervisor/environment_tools.py) and [relocation/publication tests](../tests/test_environment_tools.py)
- [Environment lifecycle and Desktop bootstrap](environments-and-updates.md)
