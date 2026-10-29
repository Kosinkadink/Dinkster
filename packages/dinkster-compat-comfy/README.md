# dinkster-compat-comfy

`dinkster-compat-comfy` is Dinkster's quarantined ComfyUI compatibility layer. It
translates ComfyUI node classes and calling conventions behind the ordinary
Dinkster worker boundary, keeping legacy behavior out of the engine and native
protocol.

## Setup

This package is a uv workspace member. From the repository root, install the
whole workspace with:

```console
uv sync --all-packages
```

The package is not published separately yet. Without a ComfyUI installation,
`dinkster-serve` mounts the provider from the separate `dinkster-native` package. Run
it in a Dinkster inference environment with the `dinkster-inference-torch[torch]`
dependencies installed, or select that interpreter with `--execution-python PATH`.
The torch-free host environment alone does not supply a sampling runtime.
Running untranslated legacy nodes requires a ComfyUI installation selected
through `--comfy-root`; `--legacy-pack` refuses without it.

The no-root provider omits implementations that still call ComfyUI modules,
including geometry/background-removal loaders and mesh postprocessing. Their
schemas are not advertised as executable. The root-backed provider retains them;
native geometry-to-FOV, mask preview, and Trellis implementations are unaffected.
`native_catalog.py` identifies these implementation dependencies, not model-family
admission rules. The generated `dinkster-native/dinkster-pack.toml` keeps execution
claims and arm declarations consistent. The schema owner retains its import
metadata and alias records; the host permits these specific schemas to have no
executor and withholds them from the executable catalog until a provider is
configured. Regenerate the native manifest after editing the source manifest or
catalog:

```console
uv run python tools/gen_native_comfy_manifests.py
```

### Pinned import schemas

`GET /api/compat/comfy/schemas` advertises upstream interfaces as import metadata,
with `nativeNodeTypes` and `importable` derived from the active native catalog.
These are not executable `comfy.*` aliases: `POST /api/compat/comfy/prompt`
lowers supported prompts to canonical native IDs before planning. An unknown
or ambiguous node refuses translation. The executable catalog stays at
`/api/nodes`. Source names resolve through declared schema aliases, without
node-specific substitutions in the alias index. EmptyLatentImage imports lower
to `dinkster.empty_latent_image`; an explicitly registered rooted legacy node keeps
its qualified `comfy.EmptyLatentImage` ID. Asset selectors accept native
digest-backed assets; legacy model filenames need configured model mounts.
SaveImage's legacy prefix uses the writable `comfy-output` mount, which can
be configured without any ComfyUI directory.

The bundled `core_schemas.json` records ComfyUI commit
`15eb748b3ec5f8a0a2d470b7fb280e2d7579f916`, the fresh master comparison pin.
It contains 642 translated core schemas and three explicit translation skips;
custom packs and partner API nodes are excluded. Regenerate from the repository
root with Python 3.12 on Linux, without a GPU:

```sh
git clone https://github.com/Comfy-Org/ComfyUI.git /tmp/dinkster-schema-reference
git -C /tmp/dinkster-schema-reference checkout 15eb748b3ec5f8a0a2d470b7fb280e2d7579f916
uv venv --python 3.12 .venv-schemas
uv pip install --python .venv-schemas/bin/python \
  --extra-index-url https://download.pytorch.org/whl/cpu \
  -r tools/comfy-core-schema-requirements.txt
.venv-schemas/bin/python tools/gen_comfy_core_schemas.py --comfyui /tmp/dinkster-schema-reference
.venv-schemas/bin/python tools/gen_comfy_core_schemas.py --comfyui /tmp/dinkster-schema-reference \
  --output /tmp/dinkster-core-schemas-second.json
cmp packages/dinkster-compat-comfy/src/dinkster_compat_comfy/core_schemas.json /tmp/dinkster-core-schemas-second.json
```

The generator requires a clean checkout at the exact pin. Updating that pin
requires inspecting schema changes and repeating regeneration; the bundled
metadata never imports or executes the reference checkout at runtime.

Dinkster's pure-stdlib packages ride the ComfyUI interpreter via `PYTHONPATH`,
which suffices for schema translation and digest *resolution*. Native nodes
that consume asset bytes (`dinkster.load_image`, `dinkster.load_checkpoint`, and
the other model loaders: `dinkster.load_lora`, `dinkster.load_lora_model_only`,
`dinkster.load_vae`, `dinkster.load_clip`, `dinkster.load_diffusion_model`) verify
content against the digest at every read (`dinkster_assets/integrity.py`), and
that requires the dependency-free `blake3` wheel in the ComfyUI venv:

```console
<selected-comfy-interpreter> -m pip install blake3
```

Before constructing compat workers, Dinkster starts that exact selected
interpreter in isolated mode and imports only `blake3`. A missing or broken
module refuses startup with the exact install command above; it never waits
for the first asset read to fail. Asset reads remain child-owned and verified
at the read site.

## Use

`translate_mappings` converts a real ComfyUI `NODE_CLASS_MAPPINGS` mapping into
a `CompatTranslation` containing Dinkster node classes and type registration:

```python
from dinkster_compat_comfy import translate_mappings

translation = translate_mappings(NODE_CLASS_MAPPINGS, only=["KSampler"])
nodes = translation.node_classes
```

Translated ComfyUI code is namespaced under `comfy.*`; legacy custom packs use
`comfy.<pack>.*`. Canonical Dinkster-native ports instead use `dinkster.*`:
`dinkster.load_image`, `dinkster.save_image`, `dinkster.load_latent`,
`dinkster.save_latent`, `dinkster.load_checkpoint`,
`dinkster.load_lora`, `dinkster.load_lora_model_only`, `dinkster.load_vae`,
`dinkster.load_clip`, `dinkster.load_diffusion_model`,
`dinkster.empty_latent_image`, `dinkster.clip_text_encode`, `dinkster.vae_decode`,
`dinkster.vae_encode`, and `dinkster.ksampler`.

`dinkster-nodes-generation` owns the universal loader, LoRA, text-encode, empty
latent, sampler, and codec schemas in that list. This pack declares their
execution bodies through `[pack] executes`; it does not own or publish those
schemas.
The boundary values are the inference-owned `dinkster.model`,
`dinkster.clip`, `dinkster.vae`, `dinkster.conditioning`, and `dinkster.latent`
contracts, plus media-owned `dinkster.image`.
This provider resolves LoRA `auto` mode to `precalculate`; `attach` is refused
because canonical conditioning carries encoded values rather than source prompts.
Its universal sampler body consumes basic single-stream conditioning; richer or
multi-stream conditioning requires a family provider.

Native latent files use `.latent` safetensors with ordered role-labeled stream
descriptors. Save preserves tensor dtype, rank, shape, and VAE source guidance;
Load also accepts stock ComfyUI files containing exactly `latent_tensor` and
the optional zero-length F32 `[0]` `latent_format_version_0` marker. It applies
the legacy `1 / 0.18215` multiplier only when that marker is absent; unrelated
safetensors metadata does not select the profile. Serialization and tensor
materialization live in `dinkster_inference_torch.latent_assets`; allocation-free
validation lives in `dinkster_assets.latent_format`. `LoadLatent` and
`comfy.LoadLatent` lower to `dinkster.load_latent`, renaming `latent` to `asset`
and preserving output 0 as native `samples`. The additional `vae_hint` output
is informational. Legacy filenames resolve exactly on the `comfy-output`
mount; loading never uses pickle or `torch.load`.

Every replacement native schema claims its legacy ComfyUI class name
(`LoadImage`, `SaveImage`, `CheckpointLoaderSimple`, `LoraLoader`,
`LoraLoaderModelOnly`, `VAELoader`, `CLIPLoader`, `UNETLoader`,
`CLIPTextEncode`, `VAEDecode`, `VAEEncode`, `KSampler`) as
a resolution alias. `merge_native_nodes` evicts the translated twins claimed
by this pack or by the composed generation owner, leaving one unambiguous
canonical schema. The native `dinkster.empty_latent_image` and legacy
`comfy.EmptyLatentImage` remain separate because their outputs are the distinct
`dinkster.latent` and `comfy.LATENT` contracts.
Legacy API prompts keep working through the prompt-boundary input adapters:
LoadImage `image` filenames convert against the `comfy-input` mount, the
model-loader name inputs (`ckpt_name`, `lora_name`, `vae_name`,
`clip_name`, `unet_name`) convert to digest-backed asset references by
exact-path lookup across every ordered live root for their designated
ComfyUI category. The 38 other exact file-backed selectors in the pinned
core catalog are likewise exposed as semantic AssetWidgets; their closed
module/class/input/category registry never promotes arbitrary custom nodes
or a whole category by inference. SaveImage `filename_prefix` becomes a
structured save target. VAELoader's non-`vae/` combo arms refuse honestly as
unported rather than mis-resolving: the composite image TAEs are not
single files, `pixel_space` never touches disk, and the video TAEs -
single files, but under `vae_approx/` - would need name-keyed subtree
dispatch. A native asset input naming a video-TAE file loads fine; only
the legacy-name arm refuses.

Combo inputs use concrete `core.combo` sockets and keep their choice lists
as baked `ComboWidget` options; their execution payloads remain strings.
Combos whose vocabulary IS a filesystem listing - the very object
`folder_paths.get_filename_list("<category>")` returned during that
node's `INPUT_TYPES()`, observed by a translation probe - additionally
carry a remote route (`/api/choices/comfy.files.<category>`, refresh
button on), so the frontend re-fetches the authoritative list instead of
trusting a startup-frozen schema. True enums, copies, postprocessed
lists, and ambiguous or ungrammatical-category listings stay static.
Recognized boolean pairs (enable/disable, on/off, true/false, yes/no -
either order, case-insensitive) translate to honest `core.boolean`
inputs wearing the original strings as `BooleanWidget` toggle labels
(never applied to a verbatim listing). Execution converts booleans back
to the v1 vocabulary, and a legacy prompt still sending the raw string
passes through verbatim.

The canonical unnamespaced core V3 `CustomCombo` is the sole
`accept_all_inputs` exception. Its exact empty `choice` COMBO declaration
translates to `choice`, optional `index`, and a closed string family whose
only member names are `option1` through `option100`. Compat prompts may send
a contiguous literal `option1..N` set; the boundary adapter nests those
values into the family. Gaps, links, non-strings, noncanonical suffixes,
custom-pack lookalikes, changed declarations, and unrelated kwargs refuse.
No general open-kwargs behavior is implied.

The pack also announces combo choice lists enumerated from its own live
ComfyUI import: `comfy.samplers`, `comfy.schedulers`, the first-class prompt
inventories `comfy.files.embeddings` and `comfy.files.loras`, plus one
`comfy.files.<category>` list per other filesystem listing the probe matched.
All are served by the composed server at `/api/choices/{id}` for remote COMBO
widgets. Legacy custom packs run in separate workers, so sampler lists they
monkey-patch are not visible in this enumeration, and their combos stay fully
static (composition refuses duplicate choice ids across workers - see ROADMAP).

Runtime-pinned in-process model packs receive a task-local
`ComponentPublisher`. Published modules enroll under the compatibility
worker's native residency coordinator and remain owned by the pack until it
is removed. Their component handles support standalone operations and can
stage alongside a runtime handle from that same coordinator, keeping both
leases under one placement lock.
Model-in/model-out apply nodes retain those handles in an `ApplicationChain`.
The universal KSampler validates the complete chain before staging, then
materializes family-owned runtime kwargs while every component lease is active.

## Learn more

- [DESIGN.md section 3.8](../../DESIGN.md#38-legacy-comfyui-node-packs-a-quarantined-on-ramp-never-a-support-surface)
  defines the compatibility and porting posture.
- [DESIGN.md sections 3.10](../../DESIGN.md#310-parallel-execution-and-memory-governance)
  and [3.12](../../DESIGN.md#312-assets-files-as-identity-not-paths) cover
  governed residents and asset-native loading.
- `tests/test_compat_comfy.py`, `tests/test_compat_prompt.py`,
  `tests/test_compat_resident.py`, and `tests/test_compat_pool.py` exercise the
  translator, prompt boundary, and residency behavior.
- `tests/test_reservations.py` and `tests/test_translate_v3.py` cover memory
  planning and V3 schema translation.
