from __future__ import annotations

from typing import Any

from cuda_loader import load_fp8_extension


def fp8_native_stats() -> dict[str, int]:
    return {str(key): int(value) for key, value in load_fp8_extension().fp8_native_stats().items()}


def reset_fp8_native_stats() -> None:
    load_fp8_extension().reset_fp8_native_stats()


def fp8_e4m3_bf16_linear(input: Any, weight: Any, scale: Any) -> Any:
    return load_fp8_extension().fp8_e4m3_bf16_linear(input.contiguous(), weight.contiguous(), scale.contiguous())


def fp8_e4m3_bf16_moe_expert(
    hidden: Any,
    gate_weight: Any,
    gate_scale: Any,
    up_weight: Any,
    up_scale: Any,
    down_weight: Any,
    down_scale: Any,
) -> Any:
    return load_fp8_extension().fp8_e4m3_bf16_moe_expert(
        hidden.contiguous(),
        gate_weight.contiguous(),
        gate_scale.contiguous(),
        up_weight.contiguous(),
        up_scale.contiguous(),
        down_weight.contiguous(),
        down_scale.contiguous(),
    )


def moe_packed_local_assignments(indices: Any, scores: Any, expert_start: int, expert_end: int) -> Any:
    return load_fp8_extension().moe_packed_local_assignments(
        indices.contiguous(),
        scores.contiguous(),
        int(expert_start),
        int(expert_end),
    )


def moe_packed_local_assignment_offsets(indices: Any, scores: Any, expert_start: int, expert_end: int) -> Any:
    return load_fp8_extension().moe_packed_local_assignment_offsets(
        indices.contiguous(),
        scores.contiguous(),
        int(expert_start),
        int(expert_end),
    )


def moe_packed_local_assignment_offsets_with_experts(indices: Any, scores: Any, expert_start: int, expert_end: int) -> Any:
    return load_fp8_extension().moe_packed_local_assignment_offsets_with_experts(
        indices.contiguous(),
        scores.contiguous(),
        int(expert_start),
        int(expert_end),
    )


def moe_packed_score_scatter_add(routed: Any, packed_output: Any, packed_tokens: Any, packed_scores: Any) -> None:
    return load_fp8_extension().moe_packed_score_scatter_add(
        routed,
        packed_output.contiguous(),
        packed_tokens.contiguous(),
        packed_scores.contiguous(),
    )


def moe_grouped_dispatch_fp8_e4m3_bf16(
    packed_hidden: Any,
    packed_tokens: Any,
    packed_scores: Any,
    unique_experts: Any,
    counts: Any,
    expert_start: int,
    gate_weights: list[Any],
    gate_scales: list[Any],
    up_weights: list[Any],
    up_scales: list[Any],
    down_weights: list[Any],
    down_scales: list[Any],
    token_count: int,
) -> Any:
    return load_fp8_extension().moe_grouped_dispatch_fp8_e4m3_bf16(
        packed_hidden.contiguous(),
        packed_tokens.contiguous(),
        packed_scores.contiguous(),
        unique_experts.contiguous(),
        counts.contiguous(),
        int(expert_start),
        [tensor.contiguous() for tensor in gate_weights],
        [tensor.contiguous() for tensor in gate_scales],
        [tensor.contiguous() for tensor in up_weights],
        [tensor.contiguous() for tensor in up_scales],
        [tensor.contiguous() for tensor in down_weights],
        [tensor.contiguous() for tensor in down_scales],
        int(token_count),
    )


def moe_grouped_dispatch_offsets_fp8_e4m3_bf16(
    flat_hidden: Any,
    packed_tokens: Any,
    packed_scores: Any,
    counts: Any,
    offsets: Any,
    expert_start: int,
    gate_weights: list[Any],
    gate_scales: list[Any],
    up_weights: list[Any],
    up_scales: list[Any],
    down_weights: list[Any],
    down_scales: list[Any],
    token_count: int,
) -> Any:
    return load_fp8_extension().moe_grouped_dispatch_offsets_fp8_e4m3_bf16(
        flat_hidden.contiguous(),
        packed_tokens.contiguous(),
        packed_scores.contiguous(),
        counts.contiguous(),
        offsets.contiguous(),
        int(expert_start),
        [tensor.contiguous() for tensor in gate_weights],
        [tensor.contiguous() for tensor in gate_scales],
        [tensor.contiguous() for tensor in up_weights],
        [tensor.contiguous() for tensor in up_scales],
        [tensor.contiguous() for tensor in down_weights],
        [tensor.contiguous() for tensor in down_scales],
        int(token_count),
    )


def moe_grouped_dispatch_offsets_segmented_fp8_e4m3_bf16(
    flat_hidden: Any,
    packed_tokens: Any,
    packed_scores: Any,
    counts: Any,
    offsets: Any,
    expert_start: int,
    gate_weights: list[Any],
    gate_scales: list[Any],
    up_weights: list[Any],
    up_scales: list[Any],
    down_weights: list[Any],
    down_scales: list[Any],
    token_count: int,
) -> Any:
    return load_fp8_extension().moe_grouped_dispatch_offsets_segmented_fp8_e4m3_bf16(
        flat_hidden.contiguous(),
        packed_tokens.contiguous(),
        packed_scores.contiguous(),
        counts.contiguous(),
        offsets.contiguous(),
        int(expert_start),
        [tensor.contiguous() for tensor in gate_weights],
        [tensor.contiguous() for tensor in gate_scales],
        [tensor.contiguous() for tensor in up_weights],
        [tensor.contiguous() for tensor in up_scales],
        [tensor.contiguous() for tensor in down_weights],
        [tensor.contiguous() for tensor in down_scales],
        int(token_count),
    )


def moe_grouped_dispatch_offsets_assignment_fp8_e4m3_bf16(
    flat_hidden: Any,
    packed_tokens: Any,
    packed_scores: Any,
    counts: Any,
    offsets: Any,
    packed_local_experts: Any,
    expert_start: int,
    gate_weights: list[Any],
    gate_scales: list[Any],
    up_weights: list[Any],
    up_scales: list[Any],
    down_weights: list[Any],
    down_scales: list[Any],
    token_count: int,
) -> Any:
    return load_fp8_extension().moe_grouped_dispatch_offsets_assignment_fp8_e4m3_bf16(
        flat_hidden.contiguous(),
        packed_tokens.contiguous(),
        packed_scores.contiguous(),
        counts.contiguous(),
        offsets.contiguous(),
        packed_local_experts.contiguous(),
        int(expert_start),
        [tensor.contiguous() for tensor in gate_weights],
        [tensor.contiguous() for tensor in gate_scales],
        [tensor.contiguous() for tensor in up_weights],
        [tensor.contiguous() for tensor in up_scales],
        [tensor.contiguous() for tensor in down_weights],
        [tensor.contiguous() for tensor in down_scales],
        int(token_count),
    )


def linear_attention_recurrent_core(query: Any, key: Any, value: Any, g: Any, beta: Any, initial_state: Any) -> Any:
    return load_fp8_extension().linear_attention_recurrent_core(
        query.contiguous(),
        key.contiguous(),
        value.contiguous(),
        g.contiguous(),
        beta.contiguous(),
        initial_state.contiguous(),
    )


def paged_attention_decode(
    query: Any,
    block_table: Any,
    key_blocks: Any,
    value_blocks: Any,
    sequence_length: int,
    block_size: int,
) -> Any:
    return load_fp8_extension().paged_attention_decode(
        query.contiguous(),
        block_table.contiguous(),
        key_blocks,
        value_blocks,
        int(sequence_length),
        int(block_size),
    )


def paged_attention_decode_batched(
    query: Any,
    block_tables: Any,
    key_blocks: list[Any],
    value_blocks: list[Any],
    sequence_lengths: Any,
    block_size: int,
) -> Any:
    return load_fp8_extension().paged_attention_decode_batched(
        query.contiguous(),
        block_tables.contiguous(),
        list(key_blocks),
        list(value_blocks),
        sequence_lengths.contiguous(),
        int(block_size),
    )
