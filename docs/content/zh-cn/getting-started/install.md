---
title: 安装指南
weight: 20
---

# 安装指南

## 前置依赖

### 必需依赖

在安装 FlagGems-vllm 之前先安装构建工具:

```bash
pip install -U 'scikit-build-core>=0.11' pybind11 ninja cmake
```

### FlagGems 依赖

**FlagGems-vllm 依赖 FlagGems**。当前部分 FlagGems-vllm 算子在底层执行时会直接调用 FlagGems 中的通用算子。在安装 FlagGems-vllm 之前，请通过 pip 安装 `flag-gems==5.4.0`（源为 [FlagOS PyPI 镜像源](https://resource.flagos.net/#browse/browse:flagos-pypi-hosted:flag-gems%2F5.4.0)）：

```bash
pip install flag-gems==5.4.0 --extra-index-url https://resource.flagos.net/repository/flagos-pypi-hosted/simple/
```

### PyTorch

在安装 FlagGems-vllm **之前**,先安装与目标加速器兼容的 PyTorch。FlagGems-vllm 不会自动重新安装 PyTorch。

对于使用 CUDA 12.1+ 的 NVIDIA GPU:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

其他后端请遵循硬件厂商的 PyTorch 安装说明。

## 从源码安装

### 基础安装

克隆并安装 FlagGems-vllm:

```bash
git clone https://github.com/flagos-ai/FlagGems-vllm.git
cd FlagGems-vllm
pip install .
```

### 开发安装

开发模式使用可编辑安装:

```bash
pip install --no-build-isolation -e .
```

### 包含测试依赖

运行测试和基准测试需要:

```bash
pip install --no-build-isolation -e '.[test]'
```

这会安装 pytest、numpy、scipy、cupy-cuda12x 等测试工具。

### 可选的 PPU FA3 性能测试依赖

PPU INT8 注意力 benchmark 通过 vLLM 的 `fa_version=3` 调用原生 BF16 FA3。
先准备厂商 PyTorch、SDK 和匹配的 vLLM 后端，再安装这项独立于 `.[test]` 的可选性能测试依赖。

`requirements/benchmark_thead.txt` 固定了已在 SAIL PyTorch
`2.13.0+v0.1.0.sail2.2.0`、SDK 2.2.0、vLLM 0.31.0 与 `vllm-sail` 环境验证过的官方
FA3 源码版本。激活相应环境和 SDK 后，从仓库根目录执行以下命令。环境中应已有 Git，
以及 `setuptools`、`wheel`、`packaging`、`ninja`、`einops` 等 Python 构建依赖。

```bash
: "${PPU_SDK:?Activate the matching PPU SDK before installing FA3}"
CUDA_VISIBLE_DEVICES="" PPU_VISIBLE_DEVICES="" \
FLASH_ATTENTION_DISABLE_BACKWARD=TRUE FLASH_ATTENTION_DISABLE_FP8=TRUE \
MAX_JOBS="${MAX_JOBS:-8}" \
python -m pip install --no-build-isolation --no-deps --no-cache-dir \
  --force-reinstall -r requirements/benchmark_thead.txt
```

安装时使用当前 PyTorch 和 SDK 编译官方 FP16/BF16 前向扩展，不替换厂商 PyTorch。
保留 paged KV、split-KV、PackGQA；省略本次 BF16 推理基线不使用的反向和 FP8。
`--no-cache-dir` 用于避免复用针对其他 PyTorch ABI 构建的旧 wheel。
更换 PyTorch 或 SDK 后应重新执行此步骤。首次源码构建会比安装匹配的 wheel 耗时更长。

安装完成后，在新进程中检查扩展加载和原生 kernel 注册：

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

该步骤只检查二进制加载和注册，不执行设备计算。运行现有注意力 benchmark 前，
仍需选择空闲设备。出现 `undefined symbol` 时，应针对当前环境重新编译扩展，
无需更改 benchmark 的 `fa_version` 或绕过可用性检查。

## 完整 GPU 环境搭建

使用提供的 `tools/setup.sh` 脚本进行完整环境引导:

```bash
source tools/setup.sh
```

此脚本会:
- 创建 `uv` 虚拟环境
- 安装 vLLM 和匹配的 PyTorch
- 安装 FlagGems-vllm
- 可选地将 Triton 替换为 FlagTree(如果设置 `USE_FLAGTREE=1`)
- 运行冒烟测试

### 环境变量

安装脚本支持以下变量:

- `DNN_VENDOR`: 后端硬件厂商(如 `nvidia`、`amd`、`ascend`)。默认:自动检测
- `VLLM_VERSION`: vLLM 版本。默认:最新版
- `USE_FLAGTREE`: 设置为 `1` 使用 FlagTree 编译器替代 Triton
- `CUDA_HOME`: CUDA 安装路径。默认:自动检测

AMD 后端示例:
```bash
DNN_VENDOR=amd source tools/setup.sh
```

## 与 FlagGems 和 vllm-plugin-fl 一起安装

完整的 FlagOS vLLM 插件栈需按以下顺序安装:

```bash
# 1. 安装 FlagGems (通用算子后端，5.4.0 版本)
pip install flag-gems==5.4.0 --extra-index-url https://resource.flagos.net/repository/flagos-pypi-hosted/simple/

# 2. 安装 FlagGems-vllm (vLLM 专用算子)
git clone https://github.com/flagos-ai/FlagGems-vllm.git
cd FlagGems-vllm
pip install --no-build-isolation -e .
cd ..

# 3. 安装 vllm-plugin-fl (vLLM 插件集成)
git clone https://github.com/flagos-ai/vllm-plugin-fl.git
cd vllm-plugin-fl
pip install --no-build-isolation -e .
```

如果安装了多个 vLLM 插件,选择 FlagOS 插件:

```bash
export VLLM_PLUGINS=fl
```

## 验证安装

### 导入冒烟测试

```bash
python -c "
import torch
import flaggems_vllm
from flaggems_vllm import runtime

print('torch:', torch.__version__)
print('硬件后端:', flaggems_vllm.vendor_name)
print('设备:', flaggems_vllm.device)
print('设备数量:', runtime.device.device_count)
print('grouped_topk:', callable(flaggems_vllm.grouped_topk))
"
```

### 运行冒烟测试

建议运行 pytest 收集检查以确认测试套件与模块导入正常：

```bash
cd FlagGems-vllm
PYTHONPATH=src pytest -q tests --collect-only
PYTHONPATH=src pytest -q tests --quick
```

## 故障排除

### CUDA 未找到

如果未找到 CUDA,设置 `CUDA_HOME`:

```bash
export CUDA_HOME=/usr/local/cuda
pip install --no-build-isolation -e .
```

### Triton 版本不匹配

FlagGems-vllm 需要 Triton ≥ 3.0。如果遇到导入错误,升级 Triton:

```bash
pip install -U triton
```

或切换到 FlagTree:

```bash
pip install flagtree
export USE_FLAGTREE=1
```

### 缺少测试依赖

如果测试失败并有导入错误,安装测试额外依赖:

```bash
pip install -e '.[test]'
```

## 下一步

- [基础用法](../usage/basic.md)
- [与 vllm-plugin-fl 集成](../usage/vllm-plugin.md)
- [运行测试](../testing/unittest.md)
- [运行基准测试](../performance/benchmark.md)
