# Browser editor quickstart

The `dinkster` command starts the engine and browser frontend on one loopback
origin, prepares missing or stale pack catalogs, and opens the editor. These
source-checkout steps work on Windows x64, Linux x64, and macOS Apple Silicon.

## Install

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and
[Node.js](https://nodejs.org/), install the pinned pnpm release with
`npm install --global pnpm@10.31.0`, then clone the backend and frontend beside
each other:

```sh
git clone https://github.com/Kosinkadink/Dinkster.git
git clone https://github.com/Kosinkadink/Dinkster-Frontend.git
cd Dinkster-Frontend
pnpm install --frozen-lockfile
pnpm --filter @dinkster/app build
cd ../Dinkster
uv sync --python 3.12 --all-packages --frozen
uv run dinkster setup
```

Until both repositories become public, these clone commands require a GitHub
account with access; authenticated users may instead use `gh repo clone`.

PowerShell, Command Prompt, and POSIX shells use the same commands. The setup
command creates the local library and managed-pack roots under `~/.dinkster`
(`%USERPROFILE%\.dinkster` on Windows). Set `DINKSTER_HOME` before setup and
launch to choose another writable location.

## Launch

```sh
uv run dinkster
```

The editor opens at `http://127.0.0.1:3639`. Its application, API, and event
connections all use that origin. The default pack suite includes the
foundation, media, and vision nodes. The default Dinkster install also includes
the independently installable `dinkster-collab` collaboration routes and
`dinkster-supervisor` process supervisor. The command above starts the engine
directly with collaboration enabled; peer-to-peer discovery is disabled for
the local launch.
The first launch prepares pack catalogs and normally takes 25 to 35 seconds before the editor
answers. The terminal reports each completed pack and the total preparation time. Later launches
skip catalog preparation unless pack code has changed.

Use Ctrl+C in the terminal to stop the engine. If the default port is busy,
run `uv run dinkster --port 4640`. Use `--no-browser` on a headless machine and
open the printed URL from a browser on that machine.

To put the supervisor in front of the engine, run:

```sh
uv run dinkster-supervisor -- uv run --no-sync dinkster-serve
```

An embedded or headless core installation may omit both optional packages.
From a source checkout, remove them from the environment and disable automatic
sync while launching so the default meta-package does not restore them:

```sh
uv sync --no-install-package dinkster-collab --no-install-package dinkster-supervisor
uv run --no-sync dinkster-serve
```

Installers that select capabilities explicitly can use the `dinkster[collab]`
and `dinkster[supervisor]` extras. The default `dinkster` install selects both.

![The browser editor after the one-command launch](quickstart.png)

## Generate a first image

The launcher does not download model weights. Download the SD 1.5 checkpoint
`v1-5-pruned-emaonly-fp16.safetensors` into a dedicated models folder, then add
that folder to `~/.dinkster/library/mounts.toml`:

```toml
[mounts.models]
path = "/absolute/path/to/models"
mode = "read"
priority = 0
```

On Windows, use a TOML path such as `C:/Users/name/Models`. Keep the existing
`[settings]` and `[mounts.output]` sections that `dinkster setup` created. See
[installation](install.md#model-folders) for the full format.

From the Dinkster checkout, build the execution environments and launch with
the environment for your accelerator. On Linux with an NVIDIA GPU:

```sh
./scripts/setup_envs.sh
DINKSTER_EXECUTION_PYTHON="$PWD/.venv-gpu/bin/python" uv run dinkster
```

On Windows with an NVIDIA GPU:

```powershell
./scripts/setup_envs.ps1
$env:DINKSTER_EXECUTION_PYTHON = "$PWD\.venv-gpu\Scripts\python.exe"
uv run dinkster
```

The setup scripts create `.venv-gpu` only when they detect an NVIDIA GPU. For a
CPU run on Linux or Windows, use `.venv-torch/bin/python` or
`.venv-torch\Scripts\python.exe` instead. On macOS Apple Silicon,
`./scripts/setup_envs.sh` installs MPS support in `.venv-torch`, so launch with:

```sh
DINKSTER_EXECUTION_PYTHON="$PWD/.venv-torch/bin/python" uv run dinkster
```

Keep this interpreter setting when the launcher refreshes pack catalogs. No
ComfyUI checkout or server is required.

The starter gallery opens on an empty workflow. Select **Stable Diffusion
1.5**, or open **Library > Templates** and select it there. In the template's
`Load Checkpoint` node, open **Browse**, choose the mounted checkpoint, and
select **Run**. The image appears in the preview and the output mount. If
Browse says that no mounted models are available, check the `models` path in
`mounts.toml` and restart Dinkster so it can scan the folder.

To build the same workflow manually instead:

1. Load the checkpoint with `Load Checkpoint`.
2. Enter positive and negative text in the two `CLIP Text Encode` nodes.
3. Connect `Empty Latent Image`, the model, and both conditioning outputs to
   `KSampler`.
4. Decode the sampled latent with `VAE Decode` and connect it to `Save Image`.
5. Select **Run** and wait for the image preview and saved output.

This is the default SD 1.5 graph: checkpoint loader, two text encoders, empty
latent image, sampler, VAE decoder, and image saver. No ComfyUI checkout or
server is part of the native launch.
