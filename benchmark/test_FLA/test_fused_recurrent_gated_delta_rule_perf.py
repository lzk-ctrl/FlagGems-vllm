# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import torch

import flaggems_vllm
from benchmark.base import Benchmark

try:
    from vllm.model_executor.layers.fla.ops import (
        fused_recurrent_gated_delta_rule as base_fused_recurrent_gated_delta_rule,
    )

    VLLM_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency guard
    base_fused_recurrent_gated_delta_rule = None
    VLLM_AVAILABLE = False


def rearrange_mixed_qkv(
    mixed_qkv,
    key_dim,
    value_dim,
    head_k_dim,
    head_v_dim,
    tp_size=1,
    contiguous=True,
):
    query, key, value = torch.split(
        mixed_qkv,
        [
            key_dim // tp_size,
            key_dim // tp_size,
            value_dim // tp_size,
        ],
        dim=-1,
    )
    query = query.view(1, query.shape[0], -1, head_k_dim)
    key = key.view(1, key.shape[0], -1, head_k_dim)
    value = value.view(1, value.shape[0], -1, head_v_dim)
    if contiguous:
        return query.contiguous(), key.contiguous(), value.contiguous()
    else:
        return query, key, value


class FusedRecurrentGatedDeltaRuleBenchmark(Benchmark):
    DEFAULT_DTYPES = [torch.bfloat16]

    def __init__(self, qkv_contiguous: bool, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.qkv_contiguous = qkv_contiguous

    def set_more_shapes(self):
        # Test the full set of sequence lengths we saw from the runtime prints
        return [
            (1,),
            (2,),
            (4,),
            (8,),
            (16,),
            (24,),
            (32,),
            (40,),
            (48,),
            (56,),
            (72,),
            (80,),
            (88,),
            (96,),
            (104,),
            (112,),
            (120,),
            (128,),
            (136,),
            (144,),
            (152,),
            (160,),
            (168,),
            (176,),
            (192,),
            (200,),
            (208,),
            (216,),
            (224,),
            (232,),
            (240,),
            (248,),
            (272,),
            (288,),
            (304,),
            (320,),
            (336,),
            (352,),
            (368,),
            (384,),
            (400,),
            (416,),
            (432,),
            (448,),
            (464,),
            (480,),
            (496,),
        ]

    def get_input_iter(self, cur_dtype):
        for (T,) in self.shapes:
            yield self._build_inputs(T, cur_dtype)

    def _build_inputs(self, T: int, dtype: torch.dtype):
        device = flaggems_vllm.device
        B = 1
        H, HV, K, V = 16, 32, 128, 128
        tp_size = 4
        key_dim = H * K
        value_dim = HV * V

        assert key_dim % tp_size == 0 and value_dim % tp_size == 0

        mixed_qkv_dim = (2 * key_dim + value_dim) // tp_size
        total_tokens = B * T
        mixed_qkv = torch.randn(
            (total_tokens, mixed_qkv_dim), device=device, dtype=dtype
        )

        q, k, v = rearrange_mixed_qkv(
            mixed_qkv,
            key_dim=key_dim,
            value_dim=value_dim,
            head_k_dim=K,
            head_v_dim=V,
            tp_size=tp_size,
            contiguous=self.qkv_contiguous,
        )

        HV_local = v.shape[2]
        g = torch.nn.functional.logsigmoid(
            torch.randn((B, T, HV_local), device=device, dtype=dtype)
        )
        beta = torch.rand(B, T, HV_local, device=device, dtype=dtype).sigmoid()
        cu_seqlens = torch.arange(T + 1, device=device, dtype=torch.long)
        initial_state = torch.zeros((1024, HV_local, K, V), device=device, dtype=dtype)
        ssm_state_indices = torch.zeros(T, device=device, dtype=torch.long)
        scale = 0.08838834764831845

        # positional args follow fused_recurrent_gated_delta_rule_fwd signature
        return (
            q,
            k,
            v,
            g,
            beta,
            scale,
            initial_state,
            True,
            cu_seqlens,
            ssm_state_indices,
            None,
            True,
        )


def _torch_op_wrapper(*args, **kwargs):
    if VLLM_AVAILABLE:
        return base_fused_recurrent_gated_delta_rule(*args, **kwargs)
    return flaggems_vllm.ops_recurrent_gated_delta_rule_fwd(*args, **kwargs)


@pytest.mark.fused_recurrent_gated_delta_rule_fwd
@pytest.mark.fused_recurrent_gated_delta_rule
@pytest.mark.parametrize("qkv_contiguous", [False])
def test_perf_fused_recurrent_gated_delta_rule(qkv_contiguous):
    bench = FusedRecurrentGatedDeltaRuleBenchmark(
        qkv_contiguous,
        op_name="fused_recurrent_gated_delta_rule",
        torch_op=_torch_op_wrapper,
    )
    bench.set_gems(flaggems_vllm.ops_recurrent_gated_delta_rule_fwd)
    bench.run()


def build_int8_gdn_decode_inputs(shape, dtype, reduction=128):
    from vllm.model_executor.layers.fla.ops.fused_recurrent import (
        fused_recurrent_gated_delta_rule_packed_decode,
    )

    tokens, packed, heads, columns = shape
    torch.manual_seed(42)
    device = flaggems_vllm.device
    mixed = torch.randn((tokens, packed), device=device, dtype=dtype)
    a = torch.randn((tokens, heads), device=device, dtype=dtype)
    b = torch.randn_like(a)
    A_log = torch.zeros(heads, device=device, dtype=torch.float32)
    dt_bias = torch.zeros_like(A_log)
    original = (
        torch.randn((tokens + 1, heads, columns, reduction), device=device, dtype=dtype)
        * 0.1
    )
    state, scales = flaggems_vllm.quantize_gdn_state_int8(original)
    # Start both paths from exactly the same representable state.
    reference_state = (state.float() * scales[..., None]).to(dtype)
    output = torch.empty((tokens, 1, heads, columns), device=device, dtype=dtype)
    reference_output = torch.empty_like(output)
    indices = torch.arange(1, tokens + 1, device=device, dtype=torch.int64)
    shared = (mixed, a, b, A_log, dt_bias, reduction**-0.5)
    reference_args = (*shared, reference_state, reference_output, indices, True)
    gems_args = (*shared, state, scales, output, indices, True)
    return fused_recurrent_gated_delta_rule_packed_decode, reference_args, gems_args


class Int8StateGdnDecodeBenchmark(FusedRecurrentGatedDeltaRuleBenchmark):
    DEFAULT_SHAPE_DESC = "T, qkv_dim, HV, V"
    set_shapes = Benchmark.set_shapes

    def set_more_shapes(self):
        return []

    def get_input_iter(self, dtype):
        for shape in self.shapes:
            baseline, reference_args, gems_args = build_int8_gdn_decode_inputs(
                shape, dtype
            )
            yield reference_args, gems_args


def vllm_int8_state_gdn_baseline(reference_args, gems_args):
    from vllm.model_executor.layers.fla.ops.fused_recurrent import (
        fused_recurrent_gated_delta_rule_packed_decode,
    )

    return fused_recurrent_gated_delta_rule_packed_decode(*reference_args)


def run_int8_state_gdn(reference_args, gems_args):
    return flaggems_vllm.gdn_w8a16_int8(*gems_args)


@pytest.mark.gdn_w8a16_int8
@pytest.mark.skipif(
    "gdn_w8a16_int8" not in vars(flaggems_vllm),
    reason="backend does not export INT8-state GDN",
)
def test_perf_gdn_w8a16_int8():
    bench = Int8StateGdnDecodeBenchmark(
        True,
        op_name="gdn_w8a16_int8",
        torch_op=vllm_int8_state_gdn_baseline,
        dtypes=[torch.bfloat16],
    )
    bench.set_gems(run_int8_state_gdn)
    bench.run()
