# bmm_w8a8_int8-PPU

## 算子描述

`bmm_w8a8_int8` 是 INT8 channel-wise BMM：对预量化的 activation 和 weight 做批量矩阵乘，再乘逐行 activation scale 与逐输出通道 weight scale。

设 activation 为 `A[B,M,K]`，weight 为 `W[B,K,N]`，scale 分别为 `sA[B,M]` 和 `sW[B,N]`：

```text
acc[b,m,n] = sum_k(int32(A[b,m,k]) * int32(W[b,k,n]))
C[b,m,n]   = cast_output(float32(acc[b,m,n]) * sA[b,m] * sW[b,n])
```

K 维使用 INT32 累加，缩放在 FP32 中完成，默认输出 BF16，也支持 FP16、FP32。输入量化由调用方完成。

| 项目 | 本版约定 |
| --- | --- |
| activation | signed INT8，连续 `[B,M,K]` |
| weight | signed INT8，连续 `[B,N,K]` 存储的转置视图 `[B,K,N]` |
| scale | 连续 FP32，支持 `[B,M]` / `[B,N]` 或 `[B,M,1]` / `[B,1,N]` |
| 输出 | 连续 `[B,M,N]`；支持调用方提供 `out` |
| 边界 | 支持空维度和 K=0；拒绝不支持的布局、dtype、scale 形状及可能溢出 INT32 的 K |
| 范围 | PPU 前向计算；不支持 batch 广播与反向传播；无 torch compute fallback |

## 算子 roofline 分析

### 硬件与测试环境

| 项目 | 配置 |
| --- | --- |
| 设备 | PPU-ZW810E |
| PyTorch | 厂商版 2.10.0 |
| 编译器 | FlagTree 0.7.0+ppu3.6 / Triton 3.6.0 |
| Python | 3.12 |
| vLLM | 0.24.0+empty |
| 本次测量输出 | BF16 |

### 算术强度与优化方向

按一次 INT8 乘加计 2 次整数运算，BF16 输出每元素占 2 byte。忽略缓存重复搬运，输入和 scale 各读取一次时：

```text
operations = 2 * B * M * N * K
bytes_min  = B * (M*K + K*N + 2*M*N + 4*M + 4*N)
AI_ideal   = operations / bytes_min
```

`bytes_min` 是理想数据搬运下界，不是实测 HBM 流量；实际分块执行可能重复读取输入。

| M | N | K | 理想算术强度（op/byte） |
| ---: | ---: | ---: | ---: |
| 16384 | 7168 | 1024 | 928.31 |
| 16384 | 2112 | 7168 | 2457.50 |
| 448 | 7168 | 256 | 195.67 |
| 14429 | 7168 | 1024 | 924.75 |
| 14429 | 2112 | 7168 | 2432.76 |

这些形状具有矩阵乘的高数据复用潜力，因此优先复用 PPU 矩阵计算路径、分块与异步加载；batch 较小时，也需要减少 kernel 发射和调度开销。本轮没有采集 HBM 实测带宽、INT8 峰值吞吐或 profiler 流水线计数，不能据此给出 roofline 脊点、硬件利用率或确定的瓶颈归因。

## 算子优化

本次完成一版实现，直接复用已有低精度 MM，没有从零重写计算核心，也没有进行多轮消融调优。

### 1. 对比基线与问题分析

同精度基线为未修改计算代码的 vLLM `triton_scaled_mm`：输入同为 INT8，使用相同的逐行、逐通道 scale，输出 BF16。由于该接口是二维 MM，BMM 对比适配为每个 batch 独立调用一次。

基线需要 B 次 kernel 发射；优化实现使用一个带 batch 维度的 grid，合并为一次发射。矩阵乘部分复用已有 PPU INT8 MM 的 JIT 计算体，降低重复实现和维护成本。

### 2. 实现方案

1. **复用已有 PPU INT8 MM 核心。** 调用 FlagGems `_mm_w8a8_aiu_kernel` 的 JIT 计算体，继承 AIU 异步 tile 加载、INT8 dot、分组 tile 排序和按 shape 选择 tile 的策略。这些是继承的优化能力，本轮没有重新实现。
2. **二维 grid 覆盖 tile 与 batch。** 第一维分配输出矩阵 tile，第二维对应独立 batch；batch 地址偏移使用 INT64。不同 batch 在一次 launch 中执行。
3. **完整 K 维累加后统一缩放。** 先做 INT32 累加，再进行 FP32 row/channel scale，避免引入不必要的分块反量化。vLLM 同精度基线也采用归约后缩放，因此不把这项设计单独计为相对基线的加速来源。
4. **直接写最终输出。** 支持连续 `out`，不需要额外拼接或布局转换。当前性能测量两侧都使用分配输出的 API，没有利用预分配输出给优化侧额外优势。

复用的 MM 引入于 FlagGems #5972（`021c78951389da53eacfe8779e3c909dc5ce6f47`），实测 FlagGems 提交为 `8f8fe0e0d123eafe0b1cd22d5b53b04aa4b03664`。旧版本缺少该依赖时，后端不导出新 BMM，仍可正常导入库。本版沿用已有 PPU tile 策略，没有新增 autotune 搜索，也没有修改 FlagTree。

### 3. 验证口径

| 项目 | 方法 |
| --- | --- |
| 精度基线 | vLLM 原版 INT8 `triton_scaled_mm`，逐 batch 调用 |
| 性能基线 | 同上；返回独立输出列表，不额外计入 `torch.stack` 的耗时 |
| 计时模式 | 仓库 `kernel` 模式，`triton.testing.do_bench` 中位数 |
| 测量参数 | warmup=100ms，rep=100ms；表中耗时单位为 ms |
| 计时范围 | 输入生成和量化在计时外；两侧均分配输出 |
| 加速比 | `baseline_latency / gems_latency` |
| 达标线 | 同精度基线的 0.95× |

empty 模式未提供原生 vLLM CUDA 扩展。独立测试环境通过官方 `vllm.platform_plugins` 入口选择 `UnspecifiedPlatform`，避开原生 CUDA 平台初始化；原版 Triton 基线仍通过厂商 PyTorch、FlagTree 在 PPU 执行。该配置未修改基线算法，也未安装厂商定制 vLLM；复现时需要相同平台插件配置。

### 4. 真实 shape 来源与权重

M/N/K 与 count 来自 `real_shapes_20261012/bmm_w8a8_fp8__int8/shapes.json`，对应 Kimi-K2.5。采集项为 BF16 `flag_gems.ops.bmm.bmm_out`，映射为量化 BMM 的测试形状。上游来源为 [FlagOSTune shape-config](https://github.com/Vincent-Xiao/FlagOSTune/tree/f4f65a7d779a0df3b017e9a80b75a7b9d651c639/shape-config)。

原始 batch 在汇总和原始 dump 中均为 `-`，本次按用户授权补充 batch=1、2、4 扫描。**M/N/K 是真实采集值，batch 是补充扫描值，不属于完整真实调用回放。**

| M | N | K | count |
| ---: | ---: | ---: | ---: |
| 16384 | 7168 | 1024 | 12 |
| 16384 | 2112 | 7168 | 5 |
| 448 | 7168 | 256 | 4 |
| 14429 | 7168 | 1024 | 4 |
| 14429 | 2112 | 7168 | 1 |

每个 batch 独立按原始调用次数加权，每组 5 个唯一形状、`sum(count)=26`：

```text
weighted_speedup = sum(speedup_i * count_i) / sum(count_i)
```

不把补充的三个 batch 设定当作三份真实调用次数，也不先按卡平均再平均。

## 最终结果

### 总体效果

| batch | count 加权加速比 | 最小单 shape 加速比 | 最大单 shape 加速比 | 加速比不大于 1 的项数 | sum(count) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.246534× | 1.123504× | 1.555556× | 0 | 26 |
| 2 | 1.279064× | 1.157246× | 1.689266× | 0 | 26 |
| 4 | 1.301112× | 1.174975× | 1.793421× | 0 | 26 |

15 组扫描均超过 0.95× 同精度目标，逐组加速比为 1.124×～1.793×，没有回退。该结论针对本次同精度基线，不代表已经验证 BF16 基线的 1.3× 目标。

### 精度与集成验证

- 56 个功能用例通过，覆盖多 batch、非整齐维度、三种输出精度、scale 形状、`out`、空维度、K=0 和输入契约错误。
- 15 组真实 M/N/K 扫描与 vLLM INT8 基线的 BF16 输出完全一致：最大绝对误差和 NRMS 均为 0；通过 / 跳过 / 失败为 15 / 0 / 0。
- 54 个 CI 工具单测、CI pin 检查和 `pre-commit run --all-files` 通过。
- 双卡 `--parallel 2` 测试通过，JSON 中包含全部逐 shape latency 和 speedup。同步补齐已有 mixin 的结果记录，并允许 CLI 接收 worker 传递的 `int8`；未改变设备映射或调度逻辑。

### 全部 shape 的耗时与加速比

下表是本轮最终验证的 15 组扫描，输入、精度、计时口径一致；所有结果均来自实际测量。

| B | M | N | K | count | vLLM INT8（ms） | Gems INT8（ms） | 加速比 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 16384 | 7168 | 1024 | 12 | 1.466040 | 1.197920 | 1.223821× |
| 2 | 16384 | 7168 | 1024 | 12 | 2.932700 | 2.381680 | 1.231358× |
| 4 | 16384 | 7168 | 1024 | 12 | 5.856400 | 4.747000 | 1.233706× |
| 1 | 16384 | 2112 | 7168 | 5 | 2.328080 | 2.072160 | 1.123504× |
| 2 | 16384 | 2112 | 7168 | 5 | 4.660880 | 4.027560 | 1.157246× |
| 4 | 16384 | 2112 | 7168 | 5 | 9.323580 | 7.915200 | 1.177934× |
| 1 | 448 | 7168 | 256 | 4 | 0.031360 | 0.020160 | 1.555556× |
| 2 | 448 | 7168 | 256 | 4 | 0.059800 | 0.035400 | 1.689266× |
| 4 | 448 | 7168 | 256 | 4 | 0.116680 | 0.065060 | 1.793421× |
| 1 | 14429 | 7168 | 1024 | 4 | 1.280360 | 1.078160 | 1.187542× |
| 2 | 14429 | 7168 | 1024 | 4 | 2.561000 | 2.145000 | 1.193939× |
| 4 | 14429 | 7168 | 1024 | 4 | 5.125720 | 4.283820 | 1.196530× |
| 1 | 14429 | 2112 | 7168 | 1 | 2.048760 | 1.806480 | 1.134117× |
| 2 | 14429 | 2112 | 7168 | 1 | 4.092440 | 3.527000 | 1.160318× |
| 4 | 14429 | 2112 | 7168 | 1 | 8.184100 | 6.965340 | 1.174975× |

### 复现命令与适用范围

在相同 PPU 软件依赖和 empty-mode 平台插件配置下，将实际 shape 写入仓库支持的 YAML，通过以下命令测量：

```bash
python -m pytest -q tests/test_bmm_w8a8_int8.py --quick
python -m pytest -q benchmark/test_bmm_w8a8_int8.py -m bmm_w8a8_int8 \
  --level core --shape_file /path/to/shapes.yaml --warmup 100 --iter 100 \
  --record json --output benchmark_bmm_w8a8_int8.json \
  --continue-on-collection-errors
```

运行前重新检查卡占用，只使用空闲卡；多个进程使用互不相交的 shape 分片。并行调度脚本与原始日志保留在测试机，不放进算子 benchmark。

本版尚未做逐项优化消融，因此不能分别量化异步加载、tile 策略和 batch 合并各自贡献；本轮没有额外 profiler 轨迹。效果结论限于上述 PPU、布局及形状扫描，未覆盖其他设备、完整 GPU 回归或模型推理。
