from __future__ import annotations

from typing import Any

from cuda_loader import load_fp8_extension


def fp8_e4m3_bf16_linear(input: Any, weight: Any, scale: Any) -> Any:
    return load_fp8_extension().fp8_e4m3_bf16_linear(input.contiguous(), weight.contiguous(), scale.contiguous())
