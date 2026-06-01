#include "fp8_linear.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fp8_e4m3_bf16_linear", &fp8_e4m3_bf16_linear, "Qwen FP8 E4M3 + BF16 block-scale linear");
    m.def("fp8_e4m3_bf16_moe_expert", &fp8_e4m3_bf16_moe_expert, "Qwen FP8 E4M3 + BF16 block-scale fused MoE expert");
    m.def("moe_packed_local_assignments", &moe_packed_local_assignments, "Qwen packed MoE local assignment planner");
    m.def("linear_attention_recurrent_core", &linear_attention_recurrent_core, "Qwen linear-attention recurrent gated-delta core");
}
