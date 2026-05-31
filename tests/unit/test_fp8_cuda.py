from __future__ import annotations

import pytest
import torch

from reference_ops import linear, silu_mul


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for the FP8 extension")
def test_fp8_cuda_linear_matches_reference_dequant_matvec_and_cublas_paths() -> None:
    try:
        from fp8_cuda import fp8_e4m3_bf16_linear
    except RuntimeError as exc:
        pytest.skip(str(exc))

    device = "cuda:1" if torch.cuda.device_count() > 1 else "cuda:0"
    torch.manual_seed(0)
    weight = (torch.randn((384, 256), device=device, dtype=torch.float32) * 0.05).to(torch.float8_e4m3fn)
    scale = torch.full((3, 2), 0.25, device=device, dtype=torch.bfloat16)
    for batch in (3, 17):
        x = torch.randn((batch, 256), device=device, dtype=torch.float32)
        out = fp8_e4m3_bf16_linear(x, weight, scale)
        ref = linear(x, weight, scale, use_cuda_kernel=False)
        torch.testing.assert_close(out, ref, atol=2e-3, rtol=2e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for the FP8 extension")
def test_fp8_cuda_moe_expert_matches_reference_small_groups() -> None:
    try:
        from fp8_cuda import fp8_e4m3_bf16_moe_expert
    except RuntimeError as exc:
        pytest.skip(str(exc))

    device = "cuda:1" if torch.cuda.device_count() > 1 else "cuda:0"
    torch.manual_seed(1)
    hidden_size = 256
    intermediate_size = 384
    gate_weight = (torch.randn((intermediate_size, hidden_size), device=device, dtype=torch.float32) * 0.05).to(torch.float8_e4m3fn)
    up_weight = (torch.randn((intermediate_size, hidden_size), device=device, dtype=torch.float32) * 0.05).to(torch.float8_e4m3fn)
    down_weight = (torch.randn((hidden_size, intermediate_size), device=device, dtype=torch.float32) * 0.05).to(torch.float8_e4m3fn)
    gate_scale = torch.full((3, 2), 0.25, device=device, dtype=torch.bfloat16)
    up_scale = torch.full((3, 2), 0.25, device=device, dtype=torch.bfloat16)
    down_scale = torch.full((2, 3), 0.25, device=device, dtype=torch.bfloat16)
    for batch in (1, 3, 4, 8):
        hidden = torch.randn((batch, hidden_size), device=device, dtype=torch.float32)
        out = fp8_e4m3_bf16_moe_expert(hidden, gate_weight, gate_scale, up_weight, up_scale, down_weight, down_scale)
        gate = linear(hidden, gate_weight, gate_scale, use_cuda_kernel=False)
        up = linear(hidden, up_weight, up_scale, use_cuda_kernel=False)
        ref = linear(silu_mul(gate, up), down_weight, down_scale, use_cuda_kernel=False)
        torch.testing.assert_close(out, ref, atol=5e-2, rtol=5e-2)


def test_fp8_cuda_linear_wrapper_falls_back_for_cpu() -> None:
    x = torch.tensor([[1.0, 2.0]])
    weight = torch.tensor([[1.0, 1.0], [2.0, 0.0]])
    scale = torch.tensor([[2.0]], dtype=torch.bfloat16)

    out = linear(x, weight, scale)

    torch.testing.assert_close(out, torch.tensor([[6.0, 4.0]]))
