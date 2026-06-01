#pragma once

#include <torch/extension.h>

#include <tuple>
#include <vector>

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
void moe_packed_score_scatter_add(
    torch::Tensor routed,
    torch::Tensor packed_output,
    torch::Tensor packed_tokens,
    torch::Tensor packed_scores);
torch::Tensor moe_grouped_dispatch_fp8_e4m3_bf16(
    torch::Tensor packed_hidden,
    torch::Tensor packed_tokens,
    torch::Tensor packed_scores,
    torch::Tensor unique_experts,
    torch::Tensor counts,
    int64_t expert_start,
    std::vector<torch::Tensor> gate_weights,
    std::vector<torch::Tensor> gate_scales,
    std::vector<torch::Tensor> up_weights,
    std::vector<torch::Tensor> up_scales,
    std::vector<torch::Tensor> down_weights,
    std::vector<torch::Tensor> down_scales,
    int64_t token_count);
std::tuple<torch::Tensor, torch::Tensor> linear_attention_recurrent_core(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor g,
    torch::Tensor beta,
    torch::Tensor initial_state);
