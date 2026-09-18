# comfy-angle / GLSL shader node study

Status: RESEARCHED 2026-07-30 (backend coordinator; user directive to
verify Dinkster can replicate ComfyUI's shader-node capability and the
native-GL dependency class it represents). Sources: Comfy-Org/
comfy-angle main, ComfyUI comfy_extras/nodes_glsl.py (present at BOTH
pins 947c2749 and e651b7be; added upstream in #12148, reworked onto
ANGLE in #13195 / CORE-162; ~846 lines at master).

## What comfy-angle actually is

A binary-distribution wheel, nothing more: Google ANGLE's libEGL +
libGLESv2 extracted from a pinned Electron release (41.0.3) and
shipped as platform wheels (win_amd64, manylinux_2_28 x86_64/aarch64,
macosx_11_0_arm64; NO macOS x86_64, NO Windows arm64). Python API is
three path helpers: get_lib_dir/get_egl_path/get_glesv2_path. It has
no shader API, no context class, no tensor interop. All execution
logic lives in ComfyUI's nodes_glsl.py.

## How the upstream node works (portability-relevant facts)

- Import-time process mutation: preloads ANGLE via ctypes (RTLD_GLOBAL
  on unix), sets PYOPENGL_PLATFORM=egl, monkey-patches
  ctypes.util.find_library on win/mac, disables OpenGL_accelerate -
  all BEFORE importing PyOpenGL. Import order is load-bearing.
- Process-local EGL singleton (GLContext): EGL display -> ES3 config
  -> 64x64 pbuffer -> GLES3 context. Display fallback chain: default
  display -> surfaceless platform -> explicit ANGLE-Vulkan. Still
  needs host drivers (Vulkan or platform GL); comfy-angle has an open
  issue "Linux wheel requires X11 on headless systems".
- Contract is GLSL ES 3.00 (WebGL2-class): fixed fullscreen-triangle
  vertex shader; user fragment shader; up to 5 input images
  (u_image0-4), 20 float + 20 int uniforms, 10 bools, 4 curve LUTs,
  4 MRT outputs (fragColor0-3), multipass via `#pragma passes N`.
- Tensor transfer is CPU-staged both ways: .cpu().numpy() float32 ->
  glTexImage2D RGBA32F, glReadPixels -> torch.from_numpy. NO CUDA-GL
  or DLPack interop exists upstream. Outputs are CPU tensors.
- Not thread-safe: singleton without locks; assumes serialized node
  execution. Forking after EGL init is unsafe; init must happen
  after worker process spawn.

## Dinkster verdict

Replicable, and Dinkster's architecture is a BETTER fit than upstream's:

1. Compat path (near-term): GLSLShader is already in the translated
   catalog at both pins (it is one of the 584 TRANSLATED at e651b7be,
   never a skip). Execution runs in the isolated compat worker, so the
   node works iff that worker's venv carries comfy-angle + PyOpenGL
   and the module import happens post-spawn (it does: workers import
   node modules after process start). Spawned isolated workers dodge
   the fork hazard by construction. Needs an executed proof, not code.
2. Native path (when triggered): a dinkster-native shader node is a
   normal isolated pack pinning comfy-angle/PyOpenGL/numpy in
   requires - exactly the partner-pack precedent (own venv, pinned
   native deps). The import-order constraint is satisfiable in a pack
   module top-level; serialization is satisfied by per-worker
   single-threaded execution. No new engine concept is required: no
   hidden inputs, plain typed image/float/int/bool/curve ports.
3. The generalization ("things like it as we expand"): the class this
   represents is packs that mutate process-global native state
   (ctypes preloads, env vars, find_library patches) and hold
   process-lifetime native contexts. Dinkster's per-pack isolated worker
   is the right containment for the whole class. Two real risks to
   track: (a) platform wheel gaps mean a pack must be able to refuse
   cleanly at import on unsupported arches without poisoning the
   catalog; (b) headless Linux workers need EGL-capable drivers -
   the ANGLE-Vulkan fallback must be validated on the actual serve
   hosts before the capability is claimed.

## Open items (ledgered in ROADMAP)

- Executed compat proof of GLSLShader in the isolated worker on this
  host (headless X-less validation of the display fallback chain).
- Import-refusal behavior audit: what happens today when a worker's
  node module raises at import on an unsupported platform.
- Decide whether the served catalog advertises platform-conditional
  availability for such nodes (frontend-visible; coordinate before
  moving).
