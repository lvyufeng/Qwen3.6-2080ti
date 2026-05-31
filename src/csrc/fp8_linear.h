#pragma once

#include <torch/extension.h>

torch::Tensor fp8_e4m3_bf16_linear(torch::Tensor input, torch::Tensor weight, torch::Tensor scale);
torch::Tensor fp8_e4m3_bf16_moe_expert(
    torch::Tensor hidden,
    torch::Tensor gate_weight,
    torch::Tensor gate_scale,
    torch::Tensor up_weight,
    torch::Tensor up_scale,
    torch::Tensor down_weight,
    torch::Tensor down_scale);
