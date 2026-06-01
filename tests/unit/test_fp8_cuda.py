from __future__ import annotations

import pytest
import torch

from reference_ops import linear, l2_norm, silu_mul


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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for the recurrent core extension")
def test_linear_attention_recurrent_core_matches_reference() -> None:
    try:
        from fp8_cuda import linear_attention_recurrent_core
    except RuntimeError as exc:
        pytest.skip(str(exc))

    device = "cuda:1" if torch.cuda.device_count() > 1 else "cuda:0"
    torch.manual_seed(2)
    for seq_len, use_initial_state in ((1, False), (1, True), (5, False), (5, True)):
        batch = 2
        heads = 3
        key_dim = 4
        value_dim = 5
        query_raw = torch.randn((batch, seq_len, heads, key_dim), device=device, dtype=torch.float32)
        key_raw = torch.randn((batch, seq_len, heads, key_dim), device=device, dtype=torch.float32)
        value_raw = torch.randn((batch, seq_len, heads, value_dim), device=device, dtype=torch.float32)
        g_raw = -torch.rand((batch, seq_len, heads), device=device, dtype=torch.float32)
        beta_raw = torch.rand((batch, seq_len, heads), device=device, dtype=torch.float32)
        initial_state = None
        if use_initial_state:
            initial_state = torch.randn((batch, heads, key_dim, value_dim), device=device, dtype=torch.float32) * 0.1

        query, key, value, g, beta, state = _prepare_recurrent_inputs(
            query_raw, key_raw, value_raw, g_raw, beta_raw, initial_state
        )
        out, final_state = linear_attention_recurrent_core(query, key, value, g, beta, state)
        ref_out, ref_state = _torch_recurrent_core(query, key, value, g, beta, state)

        torch.testing.assert_close(out, ref_out, atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(final_state, ref_state, atol=2e-5, rtol=2e-5)


def test_fp8_cuda_linear_wrapper_falls_back_for_cpu() -> None:
    x = torch.tensor([[1.0, 2.0]])
    weight = torch.tensor([[1.0, 1.0], [2.0, 0.0]])
    scale = torch.tensor([[2.0]], dtype=torch.bfloat16)

    out = linear(x, weight, scale)

    torch.testing.assert_close(out, torch.tensor([[6.0, 4.0]]))


def _prepare_recurrent_inputs(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    query = l2_norm(query).transpose(1, 2).float().contiguous()
    key = l2_norm(key).transpose(1, 2).float().contiguous()
    value = value.transpose(1, 2).float().contiguous()
    g = g.transpose(1, 2).float().contiguous()
    beta = beta.transpose(1, 2).float().contiguous()
    query = (query * (key.shape[-1] ** -0.5)).contiguous()
    if initial_state is None:
        state = torch.zeros(
            query.shape[0], query.shape[1], key.shape[-1], value.shape[-1], device=query.device, dtype=torch.float32
        )
    else:
        state = initial_state.to(device=query.device, dtype=torch.float32).contiguous()
    return query, key, value, g, beta, state


def _torch_recurrent_core(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, heads, seq_len, _ = key.shape
    value_dim = value.shape[-1]
    state = initial_state.clone()
    output = torch.zeros(batch, heads, seq_len, value_dim, device=query.device, dtype=torch.float32)
    for index in range(seq_len):
        q_t = query[:, :, index]
        k_t = key[:, :, index]
        v_t = value[:, :, index]
        state = state * g[:, :, index].exp().unsqueeze(-1).unsqueeze(-1)
        kv_mem = (state * k_t.unsqueeze(-1)).sum(dim=-2)
        delta = (v_t - kv_mem) * beta[:, :, index].unsqueeze(-1)
        state = state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        output[:, :, index] = (state * q_t.unsqueeze(-1)).sum(dim=-2)
    return output, state
