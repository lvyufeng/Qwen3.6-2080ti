from __future__ import annotations

from typing import Any

from cuda_loader import load_fp8_extension


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


def linear_attention_recurrent_core(query: Any, key: Any, value: Any, g: Any, beta: Any, initial_state: Any) -> Any:
    return load_fp8_extension().linear_attention_recurrent_core(
        query.contiguous(),
        key.contiguous(),
        value.contiguous(),
        g.contiguous(),
        beta.contiguous(),
        initial_state.contiguous(),
    )
