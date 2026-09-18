# AMD ROCm and Intel XPU hosts

Dinkster runs AMD and Intel GPUs in isolated worker environments. ROCm and XPU
install different builds of the `torch` package and cannot share a Python
environment. On a dual-vendor host, use separate worker processes.

## Support levels

- **Verified on hardware** means the repository setup and smoke gate passed on
  the exact cell below. Model-family claims are limited to the listed runs.
- **Expected** means the same hardware passed on another operating system, but
  this exact cell has not run.
- **Unsupported** means Dinkster refuses the path or has no passing evidence.
- **Deferred** means a known validation or performance question remains open.

| Backend cell | Hardware | Driver and runtime | Torch | Level |
| --- | --- | --- | --- | --- |
| Linux ROCm | AMD Radeon RX 9070 XT, `gfx1201`, 16 GiB | Ubuntu 24.04.4, kernel 7.0.0-30-generic, in-tree amdgpu, HIP 7.14.60850 | `2.12.0+rocm7.14.0` | Verified on hardware: smoke; SD1.5, SDXL, GGUF-SDXL, and SDXL LoRA eager; SD1.5 compile; ROCm 7.14 DynamicVRAM auto route; MIOpen default policy |
| Windows ROCm | AMD Radeon PRO W7800, `gfx1100`, 32 GiB | Windows 11 Pro 25H2 build 26200.9168, AMD driver 32.0.31021.5001, HIP 7.14.60850 | `2.12.0+rocm7.14.0` | Verified on hardware: smoke; SD1.5, SDXL, GGUF-SDXL, and SDXL LoRA eager |
| Linux XPU | Intel Arc B570, 10 GiB | Ubuntu 24.04.4, kernel 7.0.0-30-generic, Level Zero driver 1.15.39122+14, Intel compute runtime 26.27.39122.14 | `2.13.0+xpu`, XPU build 20260000/runtime 20.1.0 | Verified on hardware: smoke; SD1.5, SDXL, GGUF-SDXL, and SDXL LoRA eager; SD1.5 compile; hard residency cap and classic 0.4 weight ratio |
| Windows XPU | Intel Arc B570, 10 GiB | Windows 11 Pro 25H2 build 26200.8875, Intel driver 32.0.101.8974, Level Zero runtime 1.15.39183+1 | `2.13.0+xpu`, XPU build 20260000/runtime 20.1.0 | Verified on hardware: smoke; SD1.5, SDXL, GGUF-SDXL, and SDXL LoRA eager |

There are no expected-only cells today. No other AMD or Intel hardware,
driver, operating-system, or torch combination is currently supported.
Passing the smoke gate on another cell is evidence to add that exact cell; it
does not create a family-wide claim.

## Setup and smoke gate

All commands run from the repository root and recreate the backend venv. The
last command writes a JSON report and prints a short `DINKSTER ACCELERATOR SMOKE`
receipt. `status: PASS` requires complete identity, float storage and matmul,
FP16 SDPA, memory and cache APIs, synchronization, attention and dtype policy,
and CPU-parity GGUF/INT8/FP8 dequantization. Optional INT8/FP8 storage and cast
probes remain evidence-only and are counted separately.

### Linux ROCm

Install Python 3.12 and `uv`. The user running Dinkster must have access to
`/dev/kfd` and the GPU render node, normally through the `render` and `video`
groups. Install the ROCm 7.14-compatible amdgpu driver for the exact GPU; the
pinned torch wheel supplies the user-space ROCm runtime.

```bash
sudo usermod -aG render,video "$USER"
# Log out and back in once if group membership changed.
./scripts/setup_env_rocm.sh
.venv-rocm/bin/python scripts/rocm_smoke.py --json rocm-report.json
```

### Windows ROCm

Install Python 3.12, `uv`, and a ROCm 7.14-compatible AMD driver. The verified
W7800 used driver 32.0.31021.5001. AMD lists Adrenalin 26.6.4 for supported
RDNA client hardware.

```powershell
.\scripts\setup_env_rocm.ps1
.venv-rocm\Scripts\python.exe scripts\rocm_smoke.py --json rocm-report.json
```

### Linux XPU

Install Python 3.12, `uv`, and Intel's client GPU compute packages. On Ubuntu
24.04, the verified B570 used the `kobuk-team/intel-graphics` PPA:

```bash
sudo add-apt-repository -y ppa:kobuk-team/intel-graphics
sudo apt update
sudo apt install -y libze-intel-gpu1 libze1 libze-dev intel-opencl-icd intel-ocloc
./scripts/setup_env_xpu.sh
.venv-xpu/bin/python scripts/xpu_smoke.py --json xpu-report.json
```

The user must have access to the Intel render node. `intel-ocloc` and
`libze-dev` are needed for the verified Linux `torch.compile` cell, not for
the basic smoke.

### Windows XPU

Install Python 3.12, `uv`, and Intel graphics driver 32.0.101.8801 or newer,
then run:

```powershell
.\scripts\setup_env_xpu.ps1
.venv-xpu\Scripts\python.exe scripts\xpu_smoke.py --json xpu-report.json
```

## Unsupported and deferred scope

- A ROCm and XPU torch build in one Python environment is unsupported. A
  mixed-vendor single Dinkster worker process is unsupported; separate workers
  on one host are the supported boundary.
- Concurrent mixed-vendor scheduler placement and failure-isolation evidence
  is deferred. The two Linux smoke cells pass independently in concurrent
  processes, but that does not prove a production scheduler scenario.
- XPU asynchronous offload and mixed AMD/Intel multidevice attention are
  unsupported. Ordinary XPU residency is synchronous.
- Optional xformers, Sage, and external Flash Attention are unsupported on
  these cells. Comfy Kitchen support is limited to the INT8 linear and ConvRot
  dequantization routes executed by the smoke; no attention claim is made.
- Native NVFP4 execution is NVIDIA-only. XPU native FP8 matmul is unsupported.
  The smoke gate proves storage/cast evidence and device dequantization, not a
  native quantized matmul kernel.
- ROCm Windows compile is unsupported with the pinned wheel because it has no
  working Triton package. XPU Windows compile is deferred; the B570 run lacked
  the required MSVC toolchain.
- Optional aimdo-derived Intel XPU residency is deferred and is not installed
  or required. XPU uses the landed classic residency hard-cap behavior.
- Blanket model-family support is unsupported. Only the exact eager and
  compile cells in the table are hardware-verified. Serving remains eager.
- Wan VAE performance on the historical Windows W7800 `gfx1100` cell is
  deferred under issue #734. The reported gap did not reproduce on Linux
  RX 9070 XT `gfx1201`, so neither result is generalized across AMD hardware.
- Hardware outside the table, other Linux distributions, WSL, and other
  driver or torch versions are unsupported until an exact smoke and workload
  cell is recorded.

Vendor references: [ROCm compatibility matrix](https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html),
[ROCm PyTorch setup](https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/frameworks/pytorch/install.html),
[PyTorch XPU setup](https://docs.pytorch.org/docs/2.13/notes/get_start_xpu.html),
and [Intel GPU hardware table](https://dgpu-docs.intel.com/devices/hardware-table.html).
