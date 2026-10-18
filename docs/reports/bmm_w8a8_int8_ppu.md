# PPU INT8 Channel Wise BMM Optimization Report

## Operator and scope

`bmm_w8a8_int8` computes batched matrix multiplication of prequantized signed INT8 inputs with per-row activation scales and per-output-channel weight scales:

```text
C[b,m,n] = float32(sum_k(A[b,m,k] * W[b,k,n])) * activation_scale[b,m] * weight_scale[b,n]
```

The contraction accumulates in INT32, scaling uses FP32, and the output defaults to BF16. Inputs are `A[B,M,K]` in contiguous row-major layout and `W[B,K,N]` as a transpose view of contiguous `[B,N,K]` storage. FP32 scales are contiguous `[B,M]` / `[B,N]` or `[B,M,1]` / `[B,1,N]`. FP16 and FP32 outputs and a caller-owned contiguous `out` buffer are also supported.

This first implementation targets THead PPU-ZW810E, is forward-only, and does not broadcast batches. Quantization is performed by the caller. Unsupported dtypes, layouts, scale shapes, and reductions that could overflow INT32 are rejected. Empty dimensions and zero-length reductions are handled explicitly.

## Optimization methods

1. **Reuse the existing PPU INT8 MM implementation.** The batched launcher calls the existing FlagGems PPU MM JIT body rather than duplicating its matrix multiplication logic. This inherits its AIU asynchronous tile loads, INT8 dot products, grouped tile ordering, and shape-dependent tile selection.
2. **Batch work in one launch.** A two-dimensional grid assigns output tiles and independent batches. The comparison adapter launches one vLLM scaled MM per batch.
3. **Scale once after the reduction.** The full K contraction accumulates in INT32 before FP32 row/channel scaling, avoiding repeated block-wise dequantization.
4. **Write the final output directly.** No temporary batch concatenation or layout-copy kernel is required. A preallocated output is supported, although the reported benchmark uses allocation-returning APIs on both sides.

The first version reuses existing PPU MM heuristics without adding a new autotune search or modifying FlagTree. Its dependency on the existing FlagGems PPU MM JIT signature requires revalidation when that dependency changes.

## Baseline and measurement

- Hardware: PPU-ZW810E; vendor PyTorch 2.10.0; FlagTree 0.7.0+ppu3.6 / Triton 3.6.0; Python 3.12.
- Baseline: unmodified vLLM 0.24.0+empty `triton_scaled_mm`, with INT8 inputs and the same row/channel scales, called separately for each batch. This is an adapter over the vLLM MM API, not a claimed native vLLM batched INT8 operator.
- Output: BF16 for the real-M/N/K accuracy and performance sweep.
- Timing: repository `kernel` mode, `triton.testing.do_bench`, median latency in milliseconds, 100 ms warmup and 100 ms measurement duration. Input generation and quantization are outside timing. The baseline returns independent outputs as a list so an extra `torch.stack` kernel is not charged to it.
- Speedup: baseline latency divided by implementation latency. The same-precision target is at least 0.95x.

The empty package contains no native vLLM extensions. An isolated environment uses the supported `vllm.platform_plugins` entry point to select `UnspecifiedPlatform`, avoiding CUDA platform initialization that imports unavailable extensions. The original Triton implementation is unchanged and still executes on PPU through vendor PyTorch and FlagTree. This configuration supports standalone kernel measurement; acceptance systems must reproduce the plugin configuration. Full model inference is outside scope.

## Shape provenance and weights

The five M/N/K shapes and counts come from `real_shapes_20261012/bmm_w8a8_fp8__int8/shapes.json`, collected for Kimi-K2.5. The recorded operation is BF16 `flag_gems.ops.bmm.bmm_out`, mapped to the quantized BMM workload. Source: [FlagOSTune shape collection](https://github.com/Vincent-Xiao/FlagOSTune/tree/f4f65a7d779a0df3b017e9a80b75a7b9d651c639/shape-config).

The original batch field is `-`, including in the raw dumps. Batch values 1, 2 and 4 are an explicitly authorized sweep, **not complete recorded-call replays**. Compute weighted speedup independently for each batch setting:

```text
weighted_speedup = sum(speedup_i * count_i) / sum(count_i)
```

Each setting covers five unique source shapes with `sum(count)=26`; do not treat the three batch values as additional observed model calls.

| M | N | K | Count |
| ---: | ---: | ---: | ---: |
| 16384 | 7168 | 1024 | 12 |
| 16384 | 2112 | 7168 | 5 |
| 448 | 7168 | 256 | 4 |
| 14429 | 7168 | 1024 | 4 |
| 14429 | 2112 | 7168 | 1 |

## Performance results

| Batch | Count weighted speedup | Minimum | Maximum | Passed / skipped / failed |
| ---: | ---: | ---: | ---: | --- |
| 1 | 1.247x | 1.124x | 1.556x | 5 / 0 / 0 |
| 2 | 1.279x | 1.157x | 1.689x | 5 / 0 / 0 |
| 4 | 1.301x | 1.175x | 1.793x | 5 / 0 / 0 |

All 15 measured cases exceeded the 0.95x same-precision target; the lowest speedup was 1.124x. No BF16 performance claim is made. No additional deep optimization was triggered for this first version.

| B | M | N | K | vLLM INT8 ms | Gems INT8 ms | Speedup |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 16384 | 7168 | 1024 | 1.466040 | 1.197920 | 1.224x |
| 2 | 16384 | 7168 | 1024 | 2.932700 | 2.381680 | 1.231x |
| 4 | 16384 | 7168 | 1024 | 5.856400 | 4.747000 | 1.234x |
| 1 | 16384 | 2112 | 7168 | 2.328080 | 2.072160 | 1.124x |
| 2 | 16384 | 2112 | 7168 | 4.660880 | 4.027560 | 1.157x |
| 4 | 16384 | 2112 | 7168 | 9.323580 | 7.915200 | 1.178x |
| 1 | 448 | 7168 | 256 | 0.031360 | 0.020160 | 1.556x |
| 2 | 448 | 7168 | 256 | 0.059800 | 0.035400 | 1.689x |
| 4 | 448 | 7168 | 256 | 0.116680 | 0.065060 | 1.793x |
| 1 | 14429 | 7168 | 1024 | 1.280360 | 1.078160 | 1.188x |
| 2 | 14429 | 7168 | 1024 | 2.561000 | 2.145000 | 1.194x |
| 4 | 14429 | 7168 | 1024 | 5.125720 | 4.283820 | 1.197x |
| 1 | 14429 | 2112 | 7168 | 2.048760 | 1.806480 | 1.134x |
| 2 | 14429 | 2112 | 7168 | 4.092440 | 3.527000 | 1.160x |
| 4 | 14429 | 2112 | 7168 | 8.184100 | 6.965340 | 1.175x |

Raw per-shape values are available in [the results CSV](bmm_w8a8_int8_ppu_results.csv).

## Correctness and integration checks

- 56 functional tests passed: multiple batches, irregular dimensions, three output dtypes, two scale ranks, caller-owned output, empty dimensions, zero K, and input-contract errors.
- All 15 real-M/N/K scan cases matched the vLLM INT8 BF16 outputs exactly: maximum absolute difference and NRMS were both zero.
- 54 CI tooling unit tests, `tools/check_ci_pins.py`, and `pre-commit run --all-files` passed.
- Two-card `--parallel 2` smoke passed with all three shape records containing latency and speedup in JSON.
- The operator registry, pytest marker, public exports, and JSON operator name all use `bmm_w8a8_int8`.
- The shared benchmark mixin now calls `update_result`, matching `base.Benchmark`, so serial and parallel results contain JSON details. The CLI accepts `--dtypes int8`, which parallel workers already forward. Device selection and scheduling logic were not changed.
- Production code uses `torch.empty` for allocation and tensor metadata/views; it has no torch compute fallback.

Other backends, full repository GPU regression, and model inference were not tested.

## Reproduction

Use a PPU environment with vendor PyTorch, compatible FlagTree, FlagGems exposing the PPU INT8 MM implementation, and this repository installed editable. With a working vLLM empty-mode platform configuration, run:

```bash
python -m pytest -q tests/test_bmm_w8a8_int8.py --quick
python -m pytest -q benchmark/test_bmm_w8a8_int8.py -m bmm_w8a8_int8 \
  --level core --shape_file /path/to/shapes.yaml --warmup 100 --iter 100 \
  --record json --output benchmark_bmm_w8a8_int8.json \
  --continue-on-collection-errors
python -m unittest discover -s tools -p 'test_ci_*.py'
python tools/check_ci_pins.py
pre-commit run --all-files
```

Recheck device occupancy immediately before execution and bind each process to an idle device. Use five disjoint shape YAML shards for the five source M/N/K entries, each containing batch values 1, 2 and 4. Scheduling and shard files belong in test-machine logs rather than the benchmark implementation. Example shard:

```yaml
bmm_w8a8_int8:
  shapes:
    - [1, 16384, 7168, 1024]
    - [2, 16384, 7168, 1024]
    - [4, 16384, 7168, 1024]
  shape_desc: "B, M, N, K"
```

For an empty-mode package that incorrectly selects the native CUDA platform, the measurement environment used this minimal plugin in a separate virtual environment:

```python
# empty_platform.py

def register() -> str:
    return "vllm.platforms.interface.UnspecifiedPlatform"
```

```toml
# pyproject.toml
[build-system]
requires = ["setuptools"]
build-backend = "setuptools.build_meta"
[project]
name = "ops-work-empty-platform"
version = "0.0.1"
[project.entry-points."vllm.platform_plugins"]
ops_work_empty = "empty_platform:register"
[tool.setuptools]
py-modules = ["empty_platform"]
```

Install this small plugin into that environment with `python -m pip install --no-build-isolation --no-deps /path/to/plugin`, then set `VLLM_PLUGINS=ops_work_empty`. It does not install native vLLM kernels or patch the baseline algorithm. Local measurement logs retain the baseline file hash, environment package list, GPU selection, accuracy JSON, performance JSON, and aggregation script.
