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
    for batch in (1, 3, 4, 8, 16, 32):
        hidden = torch.randn((batch, hidden_size), device=device, dtype=torch.float32)
        out = fp8_e4m3_bf16_moe_expert(hidden, gate_weight, gate_scale, up_weight, up_scale, down_weight, down_scale)
        gate = linear(hidden, gate_weight, gate_scale, use_cuda_kernel=False)
        up = linear(hidden, up_weight, up_scale, use_cuda_kernel=False)
        ref = linear(silu_mul(gate, up), down_weight, down_scale, use_cuda_kernel=False)
        torch.testing.assert_close(out, ref, atol=5e-2, rtol=5e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for the assignment planner extension")
def test_moe_packed_local_assignments_matches_reference() -> None:
    try:
        from fp8_cuda import moe_packed_local_assignments
    except RuntimeError as exc:
        pytest.skip(str(exc))

    device = "cuda:1" if torch.cuda.device_count() > 1 else "cuda:0"
    torch.manual_seed(3)
    cases = [
        (torch.tensor([[3, 4], [5, 6], [7, 8]], device=device, dtype=torch.long), torch.rand((3, 2), device=device, dtype=torch.float32), 4, 8),
        (torch.tensor([[4, 4], [5, 6], [7, 7]], device=device, dtype=torch.long), torch.rand((3, 2), device=device, dtype=torch.float32), 4, 8),
        (torch.tensor([[0, 1], [2, 3]], device=device, dtype=torch.long), torch.rand((2, 2), device=device, dtype=torch.float32), 4, 8),
        (torch.tensor([[5, 4, 5], [7, 6, 4], [3, 8, 9], [4, 4, 4]], device=device, dtype=torch.long), torch.rand((4, 3), device=device, dtype=torch.float32), 4, 8),
    ]
    for indices, scores, expert_start, expert_end in cases:
        packed_tokens, packed_scores, unique_experts, counts = moe_packed_local_assignments(indices, scores, expert_start, expert_end)
        ref_tokens, ref_scores, ref_unique_experts, ref_counts = _reference_packed_assignments(indices, scores, expert_start, expert_end)
        torch.testing.assert_close(unique_experts, ref_unique_experts)
        torch.testing.assert_close(counts, ref_counts)
        _assert_assignment_groups_match(packed_tokens, packed_scores, ref_tokens, ref_scores, counts)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for the assignment planner extension")
def test_moe_packed_local_assignments_handles_empty_local_dispatch() -> None:
    try:
        from fp8_cuda import moe_packed_local_assignments
    except RuntimeError as exc:
        pytest.skip(str(exc))

    device = "cuda:1" if torch.cuda.device_count() > 1 else "cuda:0"
    indices = torch.tensor([[0, 1], [2, 3]], device=device, dtype=torch.long)
    scores = torch.rand((2, 2), device=device, dtype=torch.float32)
    packed_tokens, packed_scores, unique_experts, counts = moe_packed_local_assignments(indices, scores, 4, 8)
    assert packed_tokens.numel() == 0
    assert packed_scores.numel() == 0
    assert unique_experts.numel() == 0
    assert counts.numel() == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for the assignment offset planner extension")
def test_moe_packed_local_assignment_offsets_matches_reference() -> None:
    try:
        from fp8_cuda import moe_packed_local_assignment_offsets
    except RuntimeError as exc:
        pytest.skip(str(exc))

    device = "cuda:1" if torch.cuda.device_count() > 1 else "cuda:0"
    torch.manual_seed(13)
    cases = [
        (torch.tensor([[3, 4], [5, 6], [7, 8]], device=device, dtype=torch.long), torch.rand((3, 2), device=device, dtype=torch.float32), 4, 8),
        (torch.tensor([[4, 4], [5, 6], [7, 7]], device=device, dtype=torch.long), torch.rand((3, 2), device=device, dtype=torch.float32), 4, 8),
        (torch.tensor([[0, 1], [2, 3]], device=device, dtype=torch.long), torch.rand((2, 2), device=device, dtype=torch.float32), 4, 8),
        (torch.tensor([[5, 4, 5], [7, 6, 4], [3, 8, 9], [4, 4, 4]], device=device, dtype=torch.long), torch.rand((4, 3), device=device, dtype=torch.float32), 4, 8),
    ]
    for indices, scores, expert_start, expert_end in cases:
        packed_tokens, packed_scores, unique_experts, counts, offsets = moe_packed_local_assignment_offsets(indices, scores, expert_start, expert_end)
        ref_tokens, ref_scores, ref_unique_experts, ref_counts = _reference_packed_assignments(indices, scores, expert_start, expert_end)
        expected_counts = torch.zeros((expert_end - expert_start,), device=device, dtype=torch.long)
        expected_counts[ref_unique_experts - expert_start] = ref_counts
        expected_offsets = torch.cumsum(expected_counts, dim=0) - expected_counts
        expected_unique = torch.arange(expert_start, expert_end, device=device, dtype=torch.long)

        torch.testing.assert_close(unique_experts, expected_unique)
        torch.testing.assert_close(counts, expected_counts)
        torch.testing.assert_close(offsets, expected_offsets)
        assert packed_tokens.numel() == indices.numel()
        assert packed_scores.numel() == scores.numel()
        _assert_assignment_offset_groups_match(packed_tokens, packed_scores, ref_tokens, ref_scores, counts, offsets)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for the scatter extension")
def test_moe_packed_score_scatter_add_matches_index_add_reference() -> None:
    try:
        from fp8_cuda import moe_packed_score_scatter_add
    except RuntimeError as exc:
        pytest.skip(str(exc))

    device = "cuda:1" if torch.cuda.device_count() > 1 else "cuda:0"
    torch.manual_seed(4)
    cases = [
        (
            torch.tensor([[1.0, -2.0, 3.0], [0.5, 1.5, -1.0], [2.0, 0.0, 1.0]], device=device),
            torch.tensor([0, 1, 2], device=device, dtype=torch.long),
            torch.tensor([0.25, 0.5, -1.0], device=device),
            3,
        ),
        (
            torch.tensor([[1.0, 2.0, 0.0], [3.0, -1.0, 2.0], [0.5, 0.5, 0.5]], device=device),
            torch.tensor([1, 1, 0], device=device, dtype=torch.long),
            torch.tensor([0.5, -0.25, 2.0], device=device),
            2,
        ),
        (
            torch.randn((31, 17), device=device, dtype=torch.float32),
            torch.randint(0, 7, (31,), device=device, dtype=torch.long),
            torch.randn((31,), device=device, dtype=torch.float32),
            7,
        ),
    ]
    for packed_output, packed_tokens, packed_scores, token_count in cases:
        routed = torch.zeros((token_count, packed_output.shape[1]), device=device, dtype=torch.float32)
        ref = torch.zeros_like(routed)
        ref.index_add_(0, packed_tokens, packed_output * packed_scores[:, None])

        moe_packed_score_scatter_add(routed, packed_output, packed_tokens, packed_scores)

        torch.testing.assert_close(routed, ref, atol=1e-6, rtol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for the grouped MoE dispatch extension")
def test_moe_grouped_dispatch_fp8_e4m3_bf16_matches_reference() -> None:
    try:
        from fp8_cuda import moe_grouped_dispatch_fp8_e4m3_bf16
    except RuntimeError as exc:
        pytest.skip(str(exc))

    device = "cuda:1" if torch.cuda.device_count() > 1 else "cuda:0"
    torch.manual_seed(5)
    hidden_size = 256
    intermediate_size = 384
    gate_weights = []
    gate_scales = []
    up_weights = []
    up_scales = []
    down_weights = []
    down_scales = []
    for seed in (11, 12, 13):
        torch.manual_seed(seed)
        gate_weights.append((torch.randn((intermediate_size, hidden_size), device=device, dtype=torch.float32) * 0.04).to(torch.float8_e4m3fn))
        up_weights.append((torch.randn((intermediate_size, hidden_size), device=device, dtype=torch.float32) * 0.04).to(torch.float8_e4m3fn))
        down_weights.append((torch.randn((hidden_size, intermediate_size), device=device, dtype=torch.float32) * 0.04).to(torch.float8_e4m3fn))
        gate_scales.append(torch.full((3, 2), 0.25, device=device, dtype=torch.bfloat16))
        up_scales.append(torch.full((3, 2), 0.25, device=device, dtype=torch.bfloat16))
        down_scales.append(torch.full((2, 3), 0.25, device=device, dtype=torch.bfloat16))

    packed_hidden = torch.randn((6, hidden_size), device=device, dtype=torch.float32)
    packed_tokens = torch.tensor([2, 0, 2, 1, 1, 3], device=device, dtype=torch.long)
    packed_scores = torch.tensor([0.5, 1.25, -0.25, 0.75, 0.5, -0.5], device=device, dtype=torch.float32)
    unique_experts = torch.tensor([4, 5, 6], device=device, dtype=torch.long)
    counts = torch.tensor([2, 1, 3], device=device, dtype=torch.long)
    expert_start = 4
    token_count = 4

    routed = moe_grouped_dispatch_fp8_e4m3_bf16(
        packed_hidden,
        packed_tokens,
        packed_scores,
        unique_experts,
        counts,
        expert_start,
        gate_weights,
        gate_scales,
        up_weights,
        up_scales,
        down_weights,
        down_scales,
        token_count,
    )

    ref = torch.zeros((token_count, hidden_size), device=device, dtype=torch.float32)
    offset = 0
    for expert_id, count in zip(unique_experts.tolist(), counts.tolist(), strict=True):
        end = offset + count
        local = expert_id - expert_start
        gate = linear(packed_hidden[offset:end], gate_weights[local], gate_scales[local], use_cuda_kernel=False)
        up = linear(packed_hidden[offset:end], up_weights[local], up_scales[local], use_cuda_kernel=False)
        out = linear(silu_mul(gate, up), down_weights[local], down_scales[local], use_cuda_kernel=False)
        ref.index_add_(0, packed_tokens[offset:end], out.float() * packed_scores[offset:end, None])
        offset = end

    torch.testing.assert_close(routed, ref, atol=5e-2, rtol=5e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for the grouped MoE offset dispatch extension")
def test_moe_grouped_dispatch_offsets_fp8_e4m3_bf16_matches_reference() -> None:
    try:
        from fp8_cuda import moe_grouped_dispatch_offsets_fp8_e4m3_bf16
    except RuntimeError as exc:
        pytest.skip(str(exc))

    device = "cuda:1" if torch.cuda.device_count() > 1 else "cuda:0"
    torch.manual_seed(15)
    hidden_size = 256
    intermediate_size = 384
    gate_weights = []
    gate_scales = []
    up_weights = []
    up_scales = []
    down_weights = []
    down_scales = []
    for seed in (21, 22, 23, 24):
        torch.manual_seed(seed)
        gate_weights.append((torch.randn((intermediate_size, hidden_size), device=device, dtype=torch.float32) * 0.04).to(torch.float8_e4m3fn))
        up_weights.append((torch.randn((intermediate_size, hidden_size), device=device, dtype=torch.float32) * 0.04).to(torch.float8_e4m3fn))
        down_weights.append((torch.randn((hidden_size, intermediate_size), device=device, dtype=torch.float32) * 0.04).to(torch.float8_e4m3fn))
        gate_scales.append(torch.full((3, 2), 0.25, device=device, dtype=torch.bfloat16))
        up_scales.append(torch.full((3, 2), 0.25, device=device, dtype=torch.bfloat16))
        down_scales.append(torch.full((2, 3), 0.25, device=device, dtype=torch.bfloat16))

    flat_hidden = torch.randn((4, hidden_size), device=device, dtype=torch.float32)
    packed_tokens = torch.tensor([2, 0, 0, 0, 2, 1, 1, 3], device=device, dtype=torch.long)
    packed_scores = torch.tensor([0.5, 1.25, 0.0, 0.0, -0.25, 0.75, 0.5, -0.5], device=device, dtype=torch.float32)
    counts = torch.tensor([2, 0, 1, 3], device=device, dtype=torch.long)
    offsets = torch.tensor([0, 2, 4, 5], device=device, dtype=torch.long)
    expert_start = 4
    token_count = 4

    routed = moe_grouped_dispatch_offsets_fp8_e4m3_bf16(
        flat_hidden,
        packed_tokens,
        packed_scores,
        counts,
        offsets,
        expert_start,
        gate_weights,
        gate_scales,
        up_weights,
        up_scales,
        down_weights,
        down_scales,
        token_count,
    )

    ref = torch.zeros((token_count, hidden_size), device=device, dtype=torch.float32)
    for local, count in enumerate(counts.tolist()):
        if count == 0:
            continue
        offset = int(offsets[local].item())
        end = offset + count
        token_slice = packed_tokens[offset:end]
        hidden = flat_hidden.index_select(0, token_slice)
        gate = linear(hidden, gate_weights[local], gate_scales[local], use_cuda_kernel=False)
        up = linear(hidden, up_weights[local], up_scales[local], use_cuda_kernel=False)
        out = linear(silu_mul(gate, up), down_weights[local], down_scales[local], use_cuda_kernel=False)
        ref.index_add_(0, token_slice, out.float() * packed_scores[offset:end, None])

    torch.testing.assert_close(routed, ref, atol=5e-2, rtol=5e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for the grouped MoE offset dispatch extension")
def test_moe_grouped_dispatch_offsets_fp8_e4m3_bf16_handles_empty_dispatch() -> None:
    try:
        from fp8_cuda import moe_grouped_dispatch_offsets_fp8_e4m3_bf16
    except RuntimeError as exc:
        pytest.skip(str(exc))

    device = "cuda:1" if torch.cuda.device_count() > 1 else "cuda:0"
    hidden_size = 256
    intermediate_size = 384
    weight = torch.zeros((intermediate_size, hidden_size), device=device, dtype=torch.float8_e4m3fn)
    down = torch.zeros((hidden_size, intermediate_size), device=device, dtype=torch.float8_e4m3fn)
    scale = torch.ones((3, 2), device=device, dtype=torch.bfloat16)
    down_scale = torch.ones((2, 3), device=device, dtype=torch.bfloat16)

    routed = moe_grouped_dispatch_offsets_fp8_e4m3_bf16(
        torch.empty((3, hidden_size), device=device, dtype=torch.float32),
        torch.empty((0,), device=device, dtype=torch.long),
        torch.empty((0,), device=device, dtype=torch.float32),
        torch.zeros((1,), device=device, dtype=torch.long),
        torch.zeros((1,), device=device, dtype=torch.long),
        0,
        [weight],
        [scale],
        [weight],
        [scale],
        [down],
        [down_scale],
        3,
    )

    torch.testing.assert_close(routed, torch.zeros((3, hidden_size), device=device, dtype=torch.float32))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for the grouped MoE dispatch extension")
def test_moe_grouped_dispatch_fp8_e4m3_bf16_handles_empty_dispatch() -> None:
    try:
        from fp8_cuda import moe_grouped_dispatch_fp8_e4m3_bf16
    except RuntimeError as exc:
        pytest.skip(str(exc))

    device = "cuda:1" if torch.cuda.device_count() > 1 else "cuda:0"
    hidden_size = 256
    intermediate_size = 384
    weight = torch.zeros((intermediate_size, hidden_size), device=device, dtype=torch.float8_e4m3fn)
    down = torch.zeros((hidden_size, intermediate_size), device=device, dtype=torch.float8_e4m3fn)
    scale = torch.ones((3, 2), device=device, dtype=torch.bfloat16)
    down_scale = torch.ones((2, 3), device=device, dtype=torch.bfloat16)

    routed = moe_grouped_dispatch_fp8_e4m3_bf16(
        torch.empty((0, hidden_size), device=device, dtype=torch.float32),
        torch.empty((0,), device=device, dtype=torch.long),
        torch.empty((0,), device=device, dtype=torch.float32),
        torch.empty((0,), device=device, dtype=torch.long),
        torch.empty((0,), device=device, dtype=torch.long),
        0,
        [weight],
        [scale],
        [weight],
        [scale],
        [down],
        [down_scale],
        3,
    )

    torch.testing.assert_close(routed, torch.zeros((3, hidden_size), device=device, dtype=torch.float32))


def test_moe_packed_score_scatter_add_wrapper_preserves_routed_storage() -> None:
    if torch.cuda.is_available():
        pytest.skip("CPU-only wrapper behavior is sufficient for this non-CUDA test")
    try:
        from fp8_cuda import moe_packed_score_scatter_add
    except RuntimeError:
        pytest.skip("extension unavailable")

    routed = torch.zeros((2, 3), dtype=torch.float32)
    with pytest.raises(RuntimeError, match="routed must be CUDA"):
        moe_packed_score_scatter_add(
            routed,
            torch.zeros((1, 3), dtype=torch.float32),
            torch.zeros((1,), dtype=torch.long),
            torch.ones((1,), dtype=torch.float32),
        )
    torch.testing.assert_close(routed, torch.zeros_like(routed))


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


def _reference_packed_assignments(
    indices: torch.Tensor,
    scores: torch.Tensor,
    expert_start: int,
    expert_end: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    token_count, top_k = indices.shape
    token_ids = torch.arange(token_count, device=indices.device)
    assignment_tokens = token_ids[:, None].expand_as(indices).reshape(-1)
    assignment_experts = indices.reshape(-1)
    assignment_scores = scores.reshape(-1)
    local_mask = (assignment_experts >= expert_start) & (assignment_experts < expert_end)
    local_tokens = assignment_tokens[local_mask]
    local_experts = assignment_experts[local_mask]
    local_scores = assignment_scores[local_mask]
    order = torch.argsort(local_experts)
    packed_tokens = local_tokens[order]
    packed_scores = local_scores[order]
    packed_experts = local_experts[order]
    unique_experts, counts = torch.unique_consecutive(packed_experts, return_counts=True)
    return packed_tokens, packed_scores, unique_experts, counts


def _assert_assignment_groups_match(
    packed_tokens: torch.Tensor,
    packed_scores: torch.Tensor,
    ref_tokens: torch.Tensor,
    ref_scores: torch.Tensor,
    counts: torch.Tensor,
) -> None:
    # The native planner groups by expert but does not guarantee a stable order
    # within a group, so compare each group's (token, score) pairs as a multiset.
    offset = 0
    for count in counts.tolist():
        end = offset + count
        got = sorted(zip(packed_tokens[offset:end].tolist(), [round(s, 6) for s in packed_scores[offset:end].tolist()]))
        ref = sorted(zip(ref_tokens[offset:end].tolist(), [round(s, 6) for s in ref_scores[offset:end].tolist()]))
        assert got == ref
        offset = end
    assert offset == int(packed_tokens.numel())


def _assert_assignment_offset_groups_match(
    packed_tokens: torch.Tensor,
    packed_scores: torch.Tensor,
    ref_tokens: torch.Tensor,
    ref_scores: torch.Tensor,
    counts: torch.Tensor,
    offsets: torch.Tensor,
) -> None:
    ref_offset = 0
    for count, offset in zip(counts.tolist(), offsets.tolist(), strict=True):
        end = offset + count
        ref_end = ref_offset + count
        got = sorted(zip(packed_tokens[offset:end].tolist(), [round(s, 6) for s in packed_scores[offset:end].tolist()]))
        ref = sorted(zip(ref_tokens[ref_offset:ref_end].tolist(), [round(s, 6) for s in ref_scores[ref_offset:ref_end].tolist()]))
        assert got == ref
        ref_offset = ref_end
    assert ref_offset == int(ref_tokens.numel())


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
