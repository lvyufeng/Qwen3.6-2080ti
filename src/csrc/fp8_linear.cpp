#include "fp8_linear.h"

#include <pybind11/stl.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fp8_e4m3_bf16_linear", &fp8_e4m3_bf16_linear, "Qwen FP8 E4M3 + BF16 block-scale linear");
    m.def("fp8_e4m3_bf16_moe_expert", &fp8_e4m3_bf16_moe_expert, "Qwen FP8 E4M3 + BF16 block-scale fused MoE expert");
    m.def("moe_packed_local_assignments", &moe_packed_local_assignments, "Qwen packed MoE local assignment planner");
    m.def("moe_packed_local_assignment_offsets", &moe_packed_local_assignment_offsets, "Qwen packed MoE local assignment offset planner");
    m.def("moe_packed_score_scatter_add", &moe_packed_score_scatter_add, "Qwen packed MoE score/scatter accumulation");
    m.def("moe_grouped_dispatch_fp8_e4m3_bf16", &moe_grouped_dispatch_fp8_e4m3_bf16, "Qwen grouped FP8 MoE dispatch");
    m.def("moe_grouped_dispatch_offsets_fp8_e4m3_bf16", &moe_grouped_dispatch_offsets_fp8_e4m3_bf16, "Qwen offset grouped FP8 MoE dispatch");
    m.def("linear_attention_recurrent_core", &linear_attention_recurrent_core, "Qwen linear-attention recurrent gated-delta core");
}
