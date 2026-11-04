---
title: Installation
weight: 20
---

# Installation Guide

## Prerequisites

### Required Dependencies

Install build tools before installing FlagGems-vllm:

```bash
pip install -U 'scikit-build-core>=0.11' pybind11 ninja cmake
```

### FlagGems Dependency

**FlagGems-vllm depends on FlagGems**. Several FlagGems-vllm operators invoke general FlagGems operators internally. Before installing FlagGems-vllm, install `flag-gems==5.4.0` via pip (hosted at [FlagOS PyPI repository](https://resource.flagos.net/#browse/browse:flagos-pypi-hosted:flag-gems%2F5.4.0)):

```bash
pip install flag-gems==5.4.0 --extra-index-url https://resource.flagos.net/repository/flagos-pypi-hosted/simple/
```

### PyTorch

Install a PyTorch build compatible with your target accelerator **before** installing FlagGems-vllm. FlagGems-vllm does not reinstall PyTorch automatically.

For NVIDIA GPUs with CUDA 12.1+:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

For other backends, follow your vendor's PyTorch installation instructions.

## Install from Source

### Basic Installation

Clone and install FlagGems-vllm:

```bash
git clone https://github.com/flagos-ai/FlagGems-vllm.git
cd FlagGems-vllm
pip install .
```

### Development Installation

For development with editable mode:

```bash
pip install --no-build-isolation -e .
```

### With Test Dependencies

To run tests and benchmarks:

```bash
pip install --no-build-isolation -e '.[test]'
```

This installs pytest, numpy, scipy, cupy-cuda12x, and other test utilities.

### Optional PPU FA3 Benchmark Dependency

The PPU INT8 attention benchmark uses native BF16 FA3 through vLLM with
`fa_version=3`. Prepare the vendor PyTorch, SDK and matching vLLM backend first;
FA3 is an optional benchmark dependency, separate from `.[test]`.

`requirements/benchmark_thead.txt` pins the official FA3 source revision tested
with SAIL PyTorch `2.13.0+v0.1.0.sail2.2.0`, SDK 2.2.0 and vLLM 0.31.0 with
`vllm-sail`. Activate that environment and its SDK before running this step from
the repository root. Git and the Python build tools `setuptools`, `wheel`,
`packaging`, `ninja` and `einops` must already be installed.

```bash
: "${PPU_SDK:?Activate the matching PPU SDK before installing FA3}"
CUDA_VISIBLE_DEVICES="" PPU_VISIBLE_DEVICES="" \
FLASH_ATTENTION_DISABLE_BACKWARD=TRUE FLASH_ATTENTION_DISABLE_FP8=TRUE \
MAX_JOBS="${MAX_JOBS:-8}" \
python -m pip install --no-build-isolation --no-deps --no-cache-dir \
  --force-reinstall -r requirements/benchmark_thead.txt
```

This builds the official FP16/BF16 forward extension against the active PyTorch
and SDK without replacing PyTorch. Paged KV, split-KV and PackGQA remain enabled.
Backward and FP8 are omitted because this baseline uses BF16 inference.
`--no-cache-dir` avoids reusing a wheel built against a different PyTorch ABI;
rerun this step after changing PyTorch or the SDK. The first source build can
take substantially longer than installing a matching wheel.

Check extension loading and native registration in a fresh process:

```bash
CUDA_VISIBLE_DEVICES="" PPU_VISIBLE_DEVICES="" python - <<'PY'
import torch
import flash_attn_3._C

for name in ("flash_attn_3::fwd", "flash_attn_3::get_scheduler_metadata"):
    if not torch._C._dispatch_has_kernel_for_dispatch_key(name, "CUDA"):
        raise RuntimeError(f"Missing native FA3 kernel: {name}")
print("PyTorch:", torch.__version__)
print("Native FA3:", flash_attn_3._C.__file__)
PY
```

This checks binary loading and registration without executing a GPU workload.
Select an idle device before running the existing attention benchmark. If an
extension reports `undefined symbol`, rebuild it against the active environment
instead of changing the benchmark's `fa_version` or skipping its availability check.

## Full GPU Environment Setup

Use the provided `tools/setup.sh` script for a complete environment bootstrap:

```bash
source tools/setup.sh
```

This script:
- Creates a `uv` virtual environment
- Installs vLLM and matching PyTorch
- Installs FlagGems-vllm
- Optionally swaps Triton for FlagTree (if `USE_FLAGTREE=1`)
- Runs smoke tests

### Environment Variables

The setup script honors these variables:

- `DNN_VENDOR`: Backend vendor (e.g., `nvidia`, `amd`, `ascend`). Default: auto-detect
- `VLLM_VERSION`: vLLM version to install. Default: latest
- `USE_FLAGTREE`: Set to `1` to use FlagTree compiler instead of Triton
- `CUDA_HOME`: CUDA installation path. Default: auto-detect

Example for AMD:
```bash
DNN_VENDOR=amd source tools/setup.sh
```

## Install with FlagGems and vllm-plugin-fl

For the full FlagOS vLLM plugin stack, install in this order:

```bash
# 1. Install FlagGems (general operator backend, version 5.4.0)
pip install flag-gems==5.4.0 --extra-index-url https://resource.flagos.net/repository/flagos-pypi-hosted/simple/

# 2. Install FlagGems-vllm (vLLM-specific operators)
git clone https://github.com/flagos-ai/FlagGems-vllm.git
cd FlagGems-vllm
pip install --no-build-isolation -e .
cd ..

# 3. Install vllm-plugin-fl (vLLM plugin integration)
git clone https://github.com/flagos-ai/vllm-plugin-fl.git
cd vllm-plugin-fl
pip install --no-build-isolation -e .
```

If multiple vLLM plugins are installed, select the FlagOS plugin:

```bash
export VLLM_PLUGINS=fl
```

## Verify Installation

### Import Smoke Test

```bash
python -c "
import torch
import flaggems_vllm
from flaggems_vllm import runtime

print('torch:', torch.__version__)
print('vendor:', flaggems_vllm.vendor_name)
print('device:', flaggems_vllm.device)
print('device count:', runtime.device.device_count)
print('grouped_topk:', callable(flaggems_vllm.grouped_topk))
"
```

### Run Smoke Tests

Verify test discovery and basic module import:

```bash
cd FlagGems-vllm
PYTHONPATH=src pytest -q tests --collect-only
PYTHONPATH=src pytest -q tests --quick
```

## Troubleshooting

### Missing CUDA

If CUDA is not found, set `CUDA_HOME`:

```bash
export CUDA_HOME=/usr/local/cuda
pip install --no-build-isolation -e .
```

### Triton Version Mismatch

FlagGems-vllm requires Triton ≥ 3.0. If you see import errors, upgrade Triton:

```bash
pip install -U triton
```

Or switch to FlagTree:

```bash
pip install flagtree
export USE_FLAGTREE=1
```

### Missing Test Dependencies

If tests fail with import errors, install test extras:

```bash
pip install -e '.[test]'
```

## Next Steps

- [Basic usage patterns](../usage/basic.md)
- [Integration with vllm-plugin-fl](../usage/vllm-plugin.md)
- [Running tests](../testing/unittest.md)
- [Running benchmarks](../performance/benchmark.md)
