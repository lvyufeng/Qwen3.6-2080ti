#pragma once

#include <torch/extension.h>

#include <tuple>

torch::Tensor fp8_e4m3_bf16_linear(torch::Tensor input, torch::Tensor weight, torch::Tensor scale);
torch::Tensor fp8_e4m3_bf16_moe_expert(
    torch::Tensor hidden,
    torch::Tensor gate_weight,
    torch::Tensor gate_scale,
    torch::Tensor up_weight,
    torch::Tensor up_scale,
    torch::Tensor down_weight,
    torch::Tensor down_scale);
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> moe_packed_local_assignments(
    torch::Tensor indices,
    torch::Tensor scores,
    int64_t expert_start,
    int64_t expert_end);
std::tuple<torch::Tensor, torch::Tensor> linear_attention_recurrent_core(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor g,
    torch::Tensor beta,
    torch::Tensor initial_state);
