# Browser editor quickstart

The source run script starts the engine and browser frontend on one loopback
origin, prepares missing or stale pack catalogs, and opens the editor. These
source-checkout steps work on Windows x64, Linux x64, and Apple Silicon with
macOS 14 or newer. Intel macOS is unsupported.

## Install

Install [Node.js 26.12 or newer within 26.x](https://nodejs.org/), including npm, and
[Git](https://git-scm.com/downloads). Clone Dinkster, then run one command:

```sh
git clone https://github.com/Kosinkadink/Dinkster.git
cd Dinkster
./run.sh
```

On Windows, replace `./run.sh` with:

```powershell
.\run.ps1
```

If PowerShell blocks local scripts, use
`powershell -NoProfile -ExecutionPolicy Bypass -File .\run.ps1`.
If uv is missing, the script installs it using the
[official installer](https://docs.astral.sh/uv/getting-started/installation/)
in `~/.local/bin` (`%USERPROFILE%\.local\bin` on Windows), without sudo,
administrator rights, or shell-profile changes. Linux needs curl or wget
for this download. The installed uv is available immediately to the script.
The script builds the pinned frontend and prepares a contributor development
generation in the sibling `.<checkout-name>-development` project, then opens the
editor. Its Python, torch and tools are separate from the checkout's uv environment.
It does not need the Desktop app or a separate frontend checkout. Windows verifies
the runtime's [SQLite mapped-shm recovery](install.md#windows-interpreter).
The first run downloads dependencies and can take several minutes.
Linux and Windows default to CPU; Apple Silicon uses `mac-arm64` with MPS.
For another host-compatible variant, run `python scripts/develop.py setup
--variant NAME --yes` explicitly and start that project with the development
commands rather than rerunning the CPU-default launcher.

The scripts create the local library under `~/.dinkster`
(`%USERPROFILE%\.dinkster` on Windows). Set `DINKSTER_HOME` before launch to
choose another writable location. They preserve model mounts and output files.
They select a live contributor engine, not published setup or standalone
installed packs. The terminal and served node catalog label this mode as
unreviewed and not exactly restorable. Dependency changes require setup again;
engine source edits need only a stop/start. For explicit live pack/library inputs
and captured generations that can be restored, see
[development packs](install.md#development-packs-from-local-folders).

## Launch

```sh
./run.sh
```

Use `.\run.ps1` on Windows. After `git pull`, run the same command: environments
are refreshed and the frontend rebuilds when its pin changes. The private
`.run/frontend` directory is script-managed; do not edit it.

The editor opens at `http://127.0.0.1:3639`. Its application, API, and event
connections all use that origin. The launcher selects no packs, so an empty
editor with zero node schemas is healthy. To run nodes, select the required
pack folders through [development setup](install.md#development-packs-from-local-folders);
pack locations do not have to be inside the engine checkout. Published setup
still selects its registry defaults separately.
The process supervisor keeps the engine running after the launcher returns.

Use `python scripts/develop.py stop --yes` to stop the engine. If the default port is busy,
run `./run.sh --port 4640` (`.\run.ps1 --port 4640` on Windows).
Use `--no-browser` on a headless machine and
open the printed URL from a browser on that machine.

For a manual source launch without the script, see
[develop from source](install.md#develop-from-source). For managed published
generations, use the environment commands rather than the source launcher.

An embedded or headless core installation may omit both optional packages.
Use a separate environment for that installation. Contributor tools and the
source launcher use the active development generation, not the checkout's
root environment.

Installers that select capabilities explicitly can use the `dinkster[collab]`
and `dinkster[supervisor]` extras. The default `dinkster` install selects both.

![The browser editor after the one-command launch](quickstart.png)

## Generate a first image

Select the official `dinkster-models`, `dinkster-conditioning`,
`dinkster-latent`, `dinkster-sampling`, and `dinkster-image` packs through
[development setup](install.md#development-packs-from-local-folders) before
opening this workflow. The launcher alone selects no packs.

Save [the SD 1.5 quickstart workflow](quickstart-sd15.json), open it in the
editor, and select **Run**. If the exact checkpoint is already in a configured
model folder, Dinkster uses it. Otherwise the missing-assets dialog asks for
consent to acquire `v1-5-pruned-emaonly-fp16.safetensors` from its pinned public
Hugging Face source. Review the source, the roughly 2.13 GB size, and the
download folder when displayed, then choose **Acquire selected and retry**.
Nothing downloads merely from opening the workflow or starting the engine.

The default shared download folder is `~/Dinkster/Models/checkpoints`
(`%USERPROFILE%\\Dinkster\\Models\\checkpoints` on Windows), not the project
library. The provider filename is preserved. After verification, submission
retries automatically. The existing consent view remains busy until acquisition
and retry finish; it does not show byte progress. **Dismiss** closes that view,
not the active operation.

To use existing ComfyUI model folders or change the shared download location,
open **Assets -> Folders -> Model management**. Add the existing folder without
moving its files, preserve read-only access when downloads are not needed, and
select a writable download default. Desktop exposes the same controls under
**Existing model folders**. See [model folders](install.md#model-folders) for
shared/project visibility and advanced manual configuration.

Other workflows can reference a model by filename rather than this quickstart's
exact digest. Their missing-assets rows offer **Review source**. Choose a
suggested source, or enter a Hugging Face or Civitai file URL, and review again.
Check the provider, fixed revision, expected hash, size, reported license and
full destination before **Download**. These transfers show byte progress and
support **Cancel transfer**; **Dismiss** only stops watching. Completion shows
the actual verified file, including any reuse or collision rename. **Run once**
then retries the captured workflow; completion alone never runs it.

If the provider requires credentials, connect that provider in Model management
or the dialog, then review again. Hugging Face sign-in cannot grant Civitai
access. Local browser sign-in is offered when configured; remote Hugging Face
uses a device code and remote Civitai uses the masked token fallback. Enter
private credentials only in those product controls, never in a workflow or
source URL. Provider account access requirements still apply.

The run scripts select the execution interpreter automatically, including
during pack-catalog preparation. No ComfyUI checkout or server is required.
CPU and Apple Silicon model sampling remains subject to the
[supported host boundaries](supported/cpu-and-apple-silicon-hosts.md).

Workflow templates remain disabled by default; the linked JSON opens directly.
To build the same graph manually on the empty canvas:

1. Load the checkpoint with `Load Checkpoint`.
2. Enter positive and negative text in the two `CLIP Text Encode` nodes.
3. Connect `Empty Latent Image`, the model, and both conditioning outputs to
   `KSampler`.
4. Decode the sampled latent with `VAE Decode` and connect it to `Save Image`.
5. Select **Run** and wait for the image preview and saved output.

In `Load Checkpoint`, use **Browse** to select the mounted checkpoint. If no
models are available, check Model management for the registered folder, its
purpose and whether this project has enabled it. An existing ComfyUI models
root can be registered directly; manual mount configuration is an advanced path
in the [install guide](install.md#advanced-manual-model-configuration).

This is the default SD 1.5 graph: checkpoint loader, two text encoders, empty
latent image, sampler, VAE decoder, and image saver. No ComfyUI checkout or
server is part of the native launch.
