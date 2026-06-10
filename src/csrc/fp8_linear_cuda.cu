#include "fp8_linear.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <tuple>
#include <vector>

namespace {

constexpr int kFp8BlockSize = 128;
constexpr int kCublasBatchThreshold = 16;

__constant__ float kFp8E4M3Lut[256];

float fp8_e4m3_value_host(uint8_t code) {
    const int sign = (code >> 7) & 0x1;
    const int exp = (code >> 3) & 0xf;
    const int mant = code & 0x7;
    float value;
    if (exp == 0) {
        value = std::ldexp(static_cast<float>(mant) * (1.0f / 8.0f), -6);
    } else {
        value = std::ldexp(1.0f + static_cast<float>(mant) * (1.0f / 8.0f), exp - 7);
    }
    return sign ? -value : value;
}

bool ensure_fp8_lut() {
    static int initialized_device = -1;
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess) return false;
    if (initialized_device == device) return true;

    float fp8[256];
    for (int i = 0; i < 256; ++i) {
        fp8[i] = fp8_e4m3_value_host(static_cast<uint8_t>(i));
    }
    if (cudaMemcpyToSymbol(kFp8E4M3Lut, fp8, sizeof(fp8)) != cudaSuccess) return false;
    initialized_device = device;
    return true;
}

__device__ __forceinline__ float bf16_bits_to_float(uint16_t bits) {
    return __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&bits));
}

__global__ void fp8_e4m3_bf16_matvec_kernel(
    const float* __restrict__ x,
    const uint8_t* __restrict__ weight,
    const uint16_t* __restrict__ scale,
    float* __restrict__ y,
    int rows,
    int cols,
    int scale_cols) {
    const int row = blockIdx.x;
    const int batch = blockIdx.y;
    if (row >= rows) return;
    const int row_block = row / kFp8BlockSize;
    const float* batch_x = x + static_cast<size_t>(batch) * cols;
    float* batch_y = y + static_cast<size_t>(batch) * rows;

    float sum = 0.0f;
    for (int col = threadIdx.x; col < cols; col += blockDim.x) {
        const int col_block = col / kFp8BlockSize;
        const uint8_t code = weight[static_cast<size_t>(row) * cols + col];
        const uint16_t scale_bits = scale[static_cast<size_t>(row_block) * scale_cols + col_block];
        sum += kFp8E4M3Lut[code] * bf16_bits_to_float(scale_bits) * batch_x[col];
    }

    extern __shared__ float scratch[];
    scratch[threadIdx.x] = sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) scratch[threadIdx.x] += scratch[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) batch_y[row] = scratch[0];
}

__global__ void fp8_weight_to_half_bf16_scale_kernel(
    const uint8_t* __restrict__ weight,
    const uint16_t* __restrict__ scale,
    __half* __restrict__ out,
    int rows,
    int cols,
    int scale_cols) {
    const int64_t total = static_cast<int64_t>(rows) * cols;
    for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < total;
         idx += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        const int row = static_cast<int>(idx / cols);
        const int col = static_cast<int>(idx - static_cast<int64_t>(row) * cols);
        const uint8_t code = weight[idx];
        const uint16_t scale_bits = scale[static_cast<size_t>(row / kFp8BlockSize) * scale_cols + col / kFp8BlockSize];
        out[idx] = __float2half_rn(kFp8E4M3Lut[code] * bf16_bits_to_float(scale_bits));
    }
}

__global__ void float_rows_to_half_kernel(const float* __restrict__ x, __half* __restrict__ out, int total) {
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += gridDim.x * blockDim.x) {
        out[idx] = __float2half_rn(x[idx]);
    }
}

__global__ void silu_mul_float_to_half_kernel(
    const float* __restrict__ gate,
    const float* __restrict__ up,
    __half* __restrict__ out,
    int64_t total) {
    for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < total;
         idx += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        const float g = gate[idx];
        const float u = up[idx];
        const float silu = g / (1.0f + expf(-g));
        out[idx] = __float2half_rn(silu * u);
    }
}

__global__ void fp8_moe_gate_up_silu_kernel(
    const float* __restrict__ hidden,
    const uint8_t* __restrict__ gate_weight,
    const uint16_t* __restrict__ gate_scale,
    const uint8_t* __restrict__ up_weight,
    const uint16_t* __restrict__ up_scale,
    float* __restrict__ activation,
    int hidden_size,
    int intermediate_size,
    int input_scale_cols) {
    const int row = blockIdx.x;
    const int batch = blockIdx.y;
    if (row >= intermediate_size) return;
    const int row_block = row / kFp8BlockSize;
    const float* batch_hidden = hidden + static_cast<size_t>(batch) * hidden_size;

    float gate_sum = 0.0f;
    float up_sum = 0.0f;
    for (int col = threadIdx.x; col < hidden_size; col += blockDim.x) {
        const int col_block = col / kFp8BlockSize;
        const size_t weight_idx = static_cast<size_t>(row) * hidden_size + col;
        const size_t scale_idx = static_cast<size_t>(row_block) * input_scale_cols + col_block;
        const float x = batch_hidden[col];
        gate_sum += kFp8E4M3Lut[gate_weight[weight_idx]] * bf16_bits_to_float(gate_scale[scale_idx]) * x;
        up_sum += kFp8E4M3Lut[up_weight[weight_idx]] * bf16_bits_to_float(up_scale[scale_idx]) * x;
    }

    extern __shared__ float scratch[];
    float* gate_scratch = scratch;
    float* up_scratch = scratch + blockDim.x;
    gate_scratch[threadIdx.x] = gate_sum;
    up_scratch[threadIdx.x] = up_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            gate_scratch[threadIdx.x] += gate_scratch[threadIdx.x + stride];
            up_scratch[threadIdx.x] += up_scratch[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        const float gate = gate_scratch[0];
        const float up = up_scratch[0];
        const float silu = gate / (1.0f + expf(-gate));
        activation[static_cast<size_t>(batch) * intermediate_size + row] = silu * up;
    }
}

__global__ void fp8_moe_down_kernel(
    const float* __restrict__ activation,
    const uint8_t* __restrict__ down_weight,
    const uint16_t* __restrict__ down_scale,
    float* __restrict__ output,
    int hidden_size,
    int intermediate_size,
    int down_scale_cols) {
    const int row = blockIdx.x;
    const int batch = blockIdx.y;
    if (row >= hidden_size) return;
    const int row_block = row / kFp8BlockSize;
    const float* batch_activation = activation + static_cast<size_t>(batch) * intermediate_size;

    float sum = 0.0f;
    for (int col = threadIdx.x; col < intermediate_size; col += blockDim.x) {
        const int col_block = col / kFp8BlockSize;
        const size_t weight_idx = static_cast<size_t>(row) * intermediate_size + col;
        const size_t scale_idx = static_cast<size_t>(row_block) * down_scale_cols + col_block;
        sum += kFp8E4M3Lut[down_weight[weight_idx]] * bf16_bits_to_float(down_scale[scale_idx]) * batch_activation[col];
    }

    extern __shared__ float scratch[];
    scratch[threadIdx.x] = sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) scratch[threadIdx.x] += scratch[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) output[static_cast<size_t>(batch) * hidden_size + row] = scratch[0];
}

__global__ void fp8_moe_segmented_gate_up_silu_kernel(
    const float* __restrict__ flat_hidden,
    const int64_t* __restrict__ packed_tokens,
    const int64_t* __restrict__ counts,
    const int64_t* __restrict__ offsets,
    const uint8_t* const* __restrict__ gate_weights,
    const uint16_t* const* __restrict__ gate_scales,
    const uint8_t* const* __restrict__ up_weights,
    const uint16_t* const* __restrict__ up_scales,
    float* __restrict__ activation,
    int64_t capacity,
    int64_t token_count,
    int hidden_size,
    int intermediate_size,
    int input_scale_cols) {
    const int row = blockIdx.x;
    const int local = blockIdx.y;
    if (row >= intermediate_size) return;
    const int64_t count = counts[local];
    const int64_t offset = offsets[local];
    if (count <= 0 || offset < 0 || offset >= capacity) return;

    const uint8_t* gate_weight = gate_weights[local];
    const uint16_t* gate_scale = gate_scales[local];
    const uint8_t* up_weight = up_weights[local];
    const uint16_t* up_scale = up_scales[local];
    const int row_block = row / kFp8BlockSize;

    extern __shared__ float scratch[];
    float* gate_scratch = scratch;
    float* up_scratch = scratch + blockDim.x;
    const int64_t raw_end = offset + count;
    const int64_t end = raw_end < capacity ? raw_end : capacity;
    for (int64_t assignment = offset; assignment < end; ++assignment) {
        const int64_t token = packed_tokens[assignment];
        if (token < 0 || token >= token_count) continue;
        const float* token_hidden = flat_hidden + static_cast<size_t>(token) * hidden_size;
        float gate_sum = 0.0f;
        float up_sum = 0.0f;
        for (int col = threadIdx.x; col < hidden_size; col += blockDim.x) {
            const int col_block = col / kFp8BlockSize;
            const size_t weight_idx = static_cast<size_t>(row) * hidden_size + col;
            const size_t scale_idx = static_cast<size_t>(row_block) * input_scale_cols + col_block;
            const float x = token_hidden[col];
            gate_sum += kFp8E4M3Lut[gate_weight[weight_idx]] * bf16_bits_to_float(gate_scale[scale_idx]) * x;
            up_sum += kFp8E4M3Lut[up_weight[weight_idx]] * bf16_bits_to_float(up_scale[scale_idx]) * x;
        }

        gate_scratch[threadIdx.x] = gate_sum;
        up_scratch[threadIdx.x] = up_sum;
        __syncthreads();

        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) {
                gate_scratch[threadIdx.x] += gate_scratch[threadIdx.x + stride];
                up_scratch[threadIdx.x] += up_scratch[threadIdx.x + stride];
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            const float gate = gate_scratch[0];
            const float up = up_scratch[0];
            const float silu = gate / (1.0f + expf(-gate));
            activation[static_cast<size_t>(assignment) * intermediate_size + row] = silu * up;
        }
        __syncthreads();
    }
}

__global__ void fp8_moe_segmented_down_scatter_kernel(
    float* __restrict__ routed,
    const float* __restrict__ activation,
    const int64_t* __restrict__ packed_tokens,
    const float* __restrict__ packed_scores,
    const int64_t* __restrict__ counts,
    const int64_t* __restrict__ offsets,
    const uint8_t* const* __restrict__ down_weights,
    const uint16_t* const* __restrict__ down_scales,
    int64_t capacity,
    int64_t token_count,
    int hidden_size,
    int intermediate_size,
    int down_scale_cols) {
    const int row = blockIdx.x;
    const int local = blockIdx.y;
    if (row >= hidden_size) return;
    const int64_t count = counts[local];
    const int64_t offset = offsets[local];
    if (count <= 0 || offset < 0 || offset >= capacity) return;

    const uint8_t* down_weight = down_weights[local];
    const uint16_t* down_scale = down_scales[local];
    const int row_block = row / kFp8BlockSize;

    extern __shared__ float scratch[];
    const int64_t raw_end = offset + count;
    const int64_t end = raw_end < capacity ? raw_end : capacity;
    for (int64_t assignment = offset; assignment < end; ++assignment) {
        const int64_t token = packed_tokens[assignment];
        if (token < 0 || token >= token_count) continue;
        const float* assignment_activation = activation + static_cast<size_t>(assignment) * intermediate_size;
        float sum = 0.0f;
        for (int col = threadIdx.x; col < intermediate_size; col += blockDim.x) {
            const int col_block = col / kFp8BlockSize;
            const size_t weight_idx = static_cast<size_t>(row) * intermediate_size + col;
            const size_t scale_idx = static_cast<size_t>(row_block) * down_scale_cols + col_block;
            sum += kFp8E4M3Lut[down_weight[weight_idx]] * bf16_bits_to_float(down_scale[scale_idx]) * assignment_activation[col];
        }

        scratch[threadIdx.x] = sum;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) scratch[threadIdx.x] += scratch[threadIdx.x + stride];
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            const float score = packed_scores[assignment];
            atomicAdd(routed + token * static_cast<int64_t>(hidden_size) + row, scratch[0] * score);
        }
        __syncthreads();
    }
}

__global__ void fp8_moe_assignment_gate_up_silu_kernel(
    const float* __restrict__ flat_hidden,
    const int64_t* __restrict__ packed_tokens,
    const int64_t* __restrict__ packed_local_experts,
    const uint8_t* const* __restrict__ gate_weights,
    const uint16_t* const* __restrict__ gate_scales,
    const uint8_t* const* __restrict__ up_weights,
    const uint16_t* const* __restrict__ up_scales,
    float* __restrict__ activation,
    int64_t capacity,
    int64_t token_count,
    int local_experts,
    int hidden_size,
    int intermediate_size,
    int input_scale_cols) {
    const int row = blockIdx.x;
    const int64_t assignment = blockIdx.y;
    if (row >= intermediate_size || assignment >= capacity) return;
    const int64_t local = packed_local_experts[assignment];
    if (local < 0 || local >= local_experts) return;
    const int64_t token = packed_tokens[assignment];
    if (token < 0 || token >= token_count) return;

    const uint8_t* gate_weight = gate_weights[local];
    const uint16_t* gate_scale = gate_scales[local];
    const uint8_t* up_weight = up_weights[local];
    const uint16_t* up_scale = up_scales[local];
    const int row_block = row / kFp8BlockSize;
    const float* token_hidden = flat_hidden + static_cast<size_t>(token) * hidden_size;

    float gate_sum = 0.0f;
    float up_sum = 0.0f;
    for (int col = threadIdx.x; col < hidden_size; col += blockDim.x) {
        const int col_block = col / kFp8BlockSize;
        const size_t weight_idx = static_cast<size_t>(row) * hidden_size + col;
        const size_t scale_idx = static_cast<size_t>(row_block) * input_scale_cols + col_block;
        const float x = token_hidden[col];
        gate_sum += kFp8E4M3Lut[gate_weight[weight_idx]] * bf16_bits_to_float(gate_scale[scale_idx]) * x;
        up_sum += kFp8E4M3Lut[up_weight[weight_idx]] * bf16_bits_to_float(up_scale[scale_idx]) * x;
    }

    extern __shared__ float scratch[];
    float* gate_scratch = scratch;
    float* up_scratch = scratch + blockDim.x;
    gate_scratch[threadIdx.x] = gate_sum;
    up_scratch[threadIdx.x] = up_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            gate_scratch[threadIdx.x] += gate_scratch[threadIdx.x + stride];
            up_scratch[threadIdx.x] += up_scratch[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        const float gate = gate_scratch[0];
        const float up = up_scratch[0];
        const float silu = gate / (1.0f + expf(-gate));
        activation[static_cast<size_t>(assignment) * intermediate_size + row] = silu * up;
    }
}

__global__ void fp8_moe_assignment_down_scatter_kernel(
    float* __restrict__ routed,
    const float* __restrict__ activation,
    const int64_t* __restrict__ packed_tokens,
    const float* __restrict__ packed_scores,
    const int64_t* __restrict__ packed_local_experts,
    const uint8_t* const* __restrict__ down_weights,
    const uint16_t* const* __restrict__ down_scales,
    int64_t capacity,
    int64_t token_count,
    int local_experts,
    int hidden_size,
    int intermediate_size,
    int down_scale_cols) {
    const int row = blockIdx.x;
    const int64_t assignment = blockIdx.y;
    if (row >= hidden_size || assignment >= capacity) return;
    const int64_t local = packed_local_experts[assignment];
    if (local < 0 || local >= local_experts) return;
    const int64_t token = packed_tokens[assignment];
    if (token < 0 || token >= token_count) return;

    const uint8_t* down_weight = down_weights[local];
    const uint16_t* down_scale = down_scales[local];
    const int row_block = row / kFp8BlockSize;
    const float* assignment_activation = activation + static_cast<size_t>(assignment) * intermediate_size;

    float sum = 0.0f;
    for (int col = threadIdx.x; col < intermediate_size; col += blockDim.x) {
        const int col_block = col / kFp8BlockSize;
        const size_t weight_idx = static_cast<size_t>(row) * intermediate_size + col;
        const size_t scale_idx = static_cast<size_t>(row_block) * down_scale_cols + col_block;
        sum += kFp8E4M3Lut[down_weight[weight_idx]] * bf16_bits_to_float(down_scale[scale_idx]) * assignment_activation[col];
    }

    extern __shared__ float scratch[];
    scratch[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) scratch[threadIdx.x] += scratch[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        const float score = packed_scores[assignment];
        atomicAdd(routed + token * static_cast<int64_t>(hidden_size) + row, scratch[0] * score);
    }
}

__global__ void moe_assignment_count_kernel(
    const int64_t* __restrict__ indices,
    int64_t* __restrict__ counts,
    int64_t total_assignments,
    int top_k,
    int64_t expert_start,
    int64_t expert_end) {
    for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < total_assignments;
         idx += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        const int64_t expert = indices[idx];
        if (expert >= expert_start && expert < expert_end) {
            atomicAdd(reinterpret_cast<unsigned long long*>(&counts[expert - expert_start]), 1ULL);
        }
    }
}

__global__ void moe_assignment_fill_kernel(
    const int64_t* __restrict__ indices,
    const float* __restrict__ scores,
    int64_t* __restrict__ cursors,
    int64_t* __restrict__ packed_tokens,
    float* __restrict__ packed_scores,
    int64_t total_assignments,
    int top_k,
    int64_t expert_start,
    int64_t expert_end) {
    for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < total_assignments;
         idx += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        const int64_t expert = indices[idx];
        if (expert >= expert_start && expert < expert_end) {
            const int64_t local = expert - expert_start;
            const int64_t out_pos = static_cast<int64_t>(atomicAdd(reinterpret_cast<unsigned long long*>(&cursors[local]), 1ULL));
            packed_tokens[out_pos] = idx / top_k;
            packed_scores[out_pos] = scores[idx];
        }
    }
}

__global__ void moe_assignment_fill_with_experts_kernel(
    const int64_t* __restrict__ indices,
    const float* __restrict__ scores,
    int64_t* __restrict__ cursors,
    int64_t* __restrict__ packed_tokens,
    float* __restrict__ packed_scores,
    int64_t* __restrict__ packed_local_experts,
    int64_t total_assignments,
    int top_k,
    int64_t expert_start,
    int64_t expert_end) {
    for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < total_assignments;
         idx += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        const int64_t expert = indices[idx];
        if (expert >= expert_start && expert < expert_end) {
            const int64_t local = expert - expert_start;
            const int64_t out_pos = static_cast<int64_t>(atomicAdd(reinterpret_cast<unsigned long long*>(&cursors[local]), 1ULL));
            packed_tokens[out_pos] = idx / top_k;
            packed_scores[out_pos] = scores[idx];
            packed_local_experts[out_pos] = local;
        }
    }
}

__global__ void moe_assignment_offsets_kernel(
    const int64_t* __restrict__ counts,
    int64_t* __restrict__ offsets,
    int64_t* __restrict__ unique_experts,
    int64_t local_experts,
    int64_t expert_start) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    int64_t running = 0;
    for (int64_t local = 0; local < local_experts; ++local) {
        offsets[local] = running;
        unique_experts[local] = expert_start + local;
        running += counts[local];
    }
}

__global__ void moe_packed_score_scatter_add_kernel(
    float* __restrict__ routed,
    const float* __restrict__ packed_output,
    const int64_t* __restrict__ packed_tokens,
    const float* __restrict__ packed_scores,
    int64_t assignments,
    int hidden_size) {
    const int64_t total = assignments * static_cast<int64_t>(hidden_size);
    for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < total;
         idx += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        const int64_t assignment = idx / hidden_size;
        const int hidden = static_cast<int>(idx - assignment * hidden_size);
        const int64_t token = packed_tokens[assignment];
        const float value = packed_output[idx] * packed_scores[assignment];
        atomicAdd(routed + token * static_cast<int64_t>(hidden_size) + hidden, value);
    }
}


__global__ void linear_attention_recurrent_core_kernel(
    const float* __restrict__ query,
    const float* __restrict__ key,
    const float* __restrict__ value,
    const float* __restrict__ g,
    const float* __restrict__ beta,
    const float* __restrict__ initial_state,
    float* __restrict__ output,
    float* __restrict__ final_state,
    int heads,
    int seq_len,
    int key_dim,
    int value_dim) {
    const int lane = blockIdx.x;
    const int value_idx = blockIdx.y;
    const int batch_idx = lane / heads;
    const int head_idx = lane - batch_idx * heads;
    const int state_base = ((batch_idx * heads + head_idx) * key_dim) * value_dim + value_idx;
    const int token_base = ((batch_idx * heads + head_idx) * seq_len);

    for (int key_idx = threadIdx.x; key_idx < key_dim; key_idx += blockDim.x) {
        const size_t state_idx = static_cast<size_t>(state_base) + static_cast<size_t>(key_idx) * value_dim;
        final_state[state_idx] = initial_state[state_idx];
    }
    __syncthreads();

    extern __shared__ float scratch[];
    for (int token = 0; token < seq_len; ++token) {
        const int token_offset = token_base + token;
        const float decay = expf(g[token_offset]);
        const float beta_value = beta[token_offset];
        const float* token_key = key + static_cast<size_t>(token_offset) * key_dim;
        const float* token_query = query + static_cast<size_t>(token_offset) * key_dim;

        float kv_sum = 0.0f;
        for (int key_idx = threadIdx.x; key_idx < key_dim; key_idx += blockDim.x) {
            const size_t state_idx = static_cast<size_t>(state_base) + static_cast<size_t>(key_idx) * value_dim;
            const float state_value = final_state[state_idx] * decay;
            final_state[state_idx] = state_value;
            kv_sum += state_value * token_key[key_idx];
        }
        scratch[threadIdx.x] = kv_sum;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) scratch[threadIdx.x] += scratch[threadIdx.x + stride];
            __syncthreads();
        }
        const float kv_mem = scratch[0];
        const float token_value = value[(static_cast<size_t>(token_offset) * value_dim) + value_idx];
        const float delta = (token_value - kv_mem) * beta_value;

        float out_sum = 0.0f;
        for (int key_idx = threadIdx.x; key_idx < key_dim; key_idx += blockDim.x) {
            const size_t state_idx = static_cast<size_t>(state_base) + static_cast<size_t>(key_idx) * value_dim;
            const float updated = final_state[state_idx] + token_key[key_idx] * delta;
            final_state[state_idx] = updated;
            out_sum += updated * token_query[key_idx];
        }
        scratch[threadIdx.x] = out_sum;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) scratch[threadIdx.x] += scratch[threadIdx.x + stride];
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            output[(static_cast<size_t>(token_offset) * value_dim) + value_idx] = scratch[0];
        }
        __syncthreads();
    }
}

struct Fp8Workspace {
    int device = -1;
    __half* d_x_half = nullptr;
    __half* d_w_half = nullptr;
    size_t x_cap = 0;
    size_t w_cap = 0;
    cublasHandle_t handle = nullptr;

    ~Fp8Workspace() { release(); }

    void release() {
        cudaFree(d_x_half);
        cudaFree(d_w_half);
        if (handle != nullptr) cublasDestroy(handle);
        d_x_half = nullptr;
        d_w_half = nullptr;
        x_cap = 0;
        w_cap = 0;
        handle = nullptr;
        device = -1;
    }

    bool ensure(size_t x_elems, size_t w_elems) {
        int current_device = 0;
        if (cudaGetDevice(&current_device) != cudaSuccess) return false;
        if (device != -1 && device != current_device) release();
        device = current_device;
        if (handle == nullptr) {
            if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) return false;
            (void)cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);
        }
        if (x_cap < x_elems) {
            cudaFree(d_x_half);
            d_x_half = nullptr;
            x_cap = 0;
            if (cudaMalloc(&d_x_half, x_elems * sizeof(__half)) != cudaSuccess) return false;
            x_cap = x_elems;
        }
        if (w_cap < w_elems) {
            cudaFree(d_w_half);
            d_w_half = nullptr;
            w_cap = 0;
            if (cudaMalloc(&d_w_half, w_elems * sizeof(__half)) != cudaSuccess) return false;
            w_cap = w_elems;
        }
        return true;
    }
};

Fp8Workspace& workspace() {
    static Fp8Workspace ws;
    return ws;
}

// Reusable FP16 buffers for the routed MoE Tensor Core expert path. Holds the
// half-precision hidden/activation tiles, the three dequantized FP16 expert
// weights, and the FP32 gate/up GEMM outputs. cuBLAS handle/math mode are shared
// with Fp8Workspace via its handle, but this keeps MoE buffers independent so a
// dense FP8 linear call between expert calls does not thrash the MoE tiles.
struct MoeTensorCoreWorkspace {
    int device = -1;
    __half* d_hidden_half = nullptr;
    __half* d_gate_w_half = nullptr;
    __half* d_up_w_half = nullptr;
    __half* d_down_w_half = nullptr;
    __half* d_activation_half = nullptr;
    float* d_gate_float = nullptr;
    float* d_up_float = nullptr;
    size_t hidden_cap = 0;
    size_t gate_w_cap = 0;
    size_t up_w_cap = 0;
    size_t down_w_cap = 0;
    size_t activation_cap = 0;
    size_t gate_float_cap = 0;
    size_t up_float_cap = 0;
    cublasHandle_t handle = nullptr;

    ~MoeTensorCoreWorkspace() { release(); }

    void release() {
        cudaFree(d_hidden_half);
        cudaFree(d_gate_w_half);
        cudaFree(d_up_w_half);
        cudaFree(d_down_w_half);
        cudaFree(d_activation_half);
        cudaFree(d_gate_float);
        cudaFree(d_up_float);
        if (handle != nullptr) cublasDestroy(handle);
        d_hidden_half = nullptr;
        d_gate_w_half = nullptr;
        d_up_w_half = nullptr;
        d_down_w_half = nullptr;
        d_activation_half = nullptr;
        d_gate_float = nullptr;
        d_up_float = nullptr;
        hidden_cap = 0;
        gate_w_cap = 0;
        up_w_cap = 0;
        down_w_cap = 0;
        activation_cap = 0;
        gate_float_cap = 0;
        up_float_cap = 0;
        handle = nullptr;
        device = -1;
    }

    static bool ensure_half(__half** ptr, size_t* cap, size_t elems) {
        if (*cap >= elems) return true;
        cudaFree(*ptr);
        *ptr = nullptr;
        *cap = 0;
        if (cudaMalloc(ptr, elems * sizeof(__half)) != cudaSuccess) return false;
        *cap = elems;
        return true;
    }

    static bool ensure_float(float** ptr, size_t* cap, size_t elems) {
        if (*cap >= elems) return true;
        cudaFree(*ptr);
        *ptr = nullptr;
        *cap = 0;
        if (cudaMalloc(ptr, elems * sizeof(float)) != cudaSuccess) return false;
        *cap = elems;
        return true;
    }

    bool ensure(size_t batch, size_t hidden_size, size_t intermediate_size) {
        int current_device = 0;
        if (cudaGetDevice(&current_device) != cudaSuccess) return false;
        if (device != -1 && device != current_device) release();
        device = current_device;
        if (handle == nullptr) {
            if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) return false;
            (void)cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);
        }
        const size_t hidden_elems = batch * hidden_size;
        const size_t activation_elems = batch * intermediate_size;
        const size_t gate_w_elems = intermediate_size * hidden_size;
        const size_t down_w_elems = hidden_size * intermediate_size;
        return ensure_half(&d_hidden_half, &hidden_cap, hidden_elems) &&
               ensure_half(&d_gate_w_half, &gate_w_cap, gate_w_elems) &&
               ensure_half(&d_up_w_half, &up_w_cap, gate_w_elems) &&
               ensure_half(&d_down_w_half, &down_w_cap, down_w_elems) &&
               ensure_half(&d_activation_half, &activation_cap, activation_elems) &&
               ensure_float(&d_gate_float, &gate_float_cap, activation_elems) &&
               ensure_float(&d_up_float, &up_float_cap, activation_elems);
    }
};

MoeTensorCoreWorkspace& moe_tensor_core_workspace() {
    static MoeTensorCoreWorkspace ws;
    return ws;
}

// Tensor Core MoE compute is enabled by default; QWEN36_NATIVE_MOE_TENSOR_CORE=0
// forces the scalar FP32 expert kernels (used for A/B benchmarking).
bool moe_tensor_core_enabled() {
    static int cached = -1;
    if (cached < 0) {
        const char* env = std::getenv("QWEN36_NATIVE_MOE_TENSOR_CORE");
        cached = (env != nullptr && std::strcmp(env, "0") == 0) ? 0 : 1;
    }
    return cached != 0;
}

// sm_70+ exposes FP16 Tensor Cores (HMMA). Turing sm_75 (RTX 2080 Ti) qualifies.
bool device_has_fp16_tensor_core() {
    static int cached_device = -1;
    static bool cached_result = false;
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess) return false;
    if (device == cached_device) return cached_result;
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, device) != cudaSuccess) return false;
    cached_device = device;
    cached_result = prop.major >= 7;
    return cached_result;
}

int moe_tensor_core_min_group_tokens() {
    static int cached = -1;
    if (cached < 0) {
        const char* env = std::getenv("QWEN36_NATIVE_MOE_TENSOR_CORE_MIN_GROUP_TOKENS");
        cached = env == nullptr ? 1 : std::max(1, std::atoi(env));
    }
    return cached;
}

void check_cuda(cudaError_t err, const char* what) {
    TORCH_CHECK(err == cudaSuccess, what, ": ", cudaGetErrorString(err));
}

void check_cublas(cublasStatus_t status, const char* what) {
    TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS, what, ": cublas status ", static_cast<int>(status));
}

void check_moe_expert_tensor_basics(const torch::Tensor& tensor, const char* name, at::ScalarType scalar_type) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be CUDA");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(tensor.scalar_type() == scalar_type, name, " has unexpected dtype");
    TORCH_CHECK(tensor.dim() == 2, name, " must be rank-2");
}

__global__ void paged_attention_decode_f32_kernel(
    const float* __restrict__ query,
    const int64_t* __restrict__ block_table,
    const float* __restrict__ key_blocks,
    const float* __restrict__ value_blocks,
    float* __restrict__ output,
    int batch,
    int heads,
    int physical_blocks,
    int block_size,
    int head_dim,
    int sequence_length) {
    const int head_index = blockIdx.x;
    const int b = head_index / heads;
    const int h = head_index - b * heads;
    if (b >= batch) return;

    extern __shared__ float shared[];
    float* reduce = shared;
    float* acc = shared + blockDim.x;
    __shared__ float max_score;
    __shared__ float weight;
    __shared__ float denom;

    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        acc[d] = 0.0f;
    }
    if (threadIdx.x == 0) max_score = -3.402823e38f;
    __syncthreads();

    const size_t query_base = (static_cast<size_t>(b) * heads + h) * head_dim;
    for (int token = 0; token < sequence_length; ++token) {
        const int logical_block = token / block_size;
        const int block_offset = token - logical_block * block_size;
        const int64_t physical_block64 = block_table[logical_block];
        const int physical_block = static_cast<int>(physical_block64);
        if (physical_block < 0 || physical_block >= physical_blocks) return;
        const size_t kv_base = ((((static_cast<size_t>(b) * heads + h) * physical_blocks + physical_block) * block_size + block_offset) * head_dim);

        float partial = 0.0f;
        for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
            partial += query[query_base + d] * key_blocks[kv_base + d];
        }
        reduce[threadIdx.x] = partial;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) reduce[threadIdx.x] += reduce[threadIdx.x + stride];
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            const float score = reduce[0] * rsqrtf(static_cast<float>(head_dim));
            max_score = fmaxf(max_score, score);
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) denom = 0.0f;
    __syncthreads();

    for (int token = 0; token < sequence_length; ++token) {
        const int logical_block = token / block_size;
        const int block_offset = token - logical_block * block_size;
        const int physical_block = static_cast<int>(block_table[logical_block]);
        const size_t kv_base = ((((static_cast<size_t>(b) * heads + h) * physical_blocks + physical_block) * block_size + block_offset) * head_dim);

        float partial = 0.0f;
        for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
            partial += query[query_base + d] * key_blocks[kv_base + d];
        }
        reduce[threadIdx.x] = partial;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) reduce[threadIdx.x] += reduce[threadIdx.x + stride];
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            const float score = reduce[0] * rsqrtf(static_cast<float>(head_dim));
            weight = expf(score - max_score);
            denom += weight;
        }
        __syncthreads();
        for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
            acc[d] += weight * value_blocks[kv_base + d];
        }
        __syncthreads();
    }

    const size_t output_base = query_base;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        output[output_base + d] = acc[d] / denom;
    }
}

__global__ void paged_attention_decode_batched_f32_kernel(
    const float* __restrict__ query,
    const int64_t* __restrict__ block_tables,
    const float* const* __restrict__ key_block_ptrs,
    const float* const* __restrict__ value_block_ptrs,
    const int64_t* __restrict__ physical_block_counts,
    const int64_t* __restrict__ sequence_lengths,
    float* __restrict__ output,
    int batch,
    int heads,
    int max_blocks,
    int block_size,
    int head_dim) {
    const int head_index = blockIdx.x;
    const int b = head_index / heads;
    const int h = head_index - b * heads;
    if (b >= batch) return;

    extern __shared__ float shared[];
    float* reduce = shared;
    float* acc = shared + blockDim.x;
    __shared__ float max_score;
    __shared__ float weight;
    __shared__ float denom;

    const int sequence_length = static_cast<int>(sequence_lengths[b]);
    const int physical_blocks = static_cast<int>(physical_block_counts[b]);
    const float* key_blocks = key_block_ptrs[b];
    const float* value_blocks = value_block_ptrs[b];
    const int64_t* block_table = block_tables + static_cast<size_t>(b) * max_blocks;
    const size_t query_base = (static_cast<size_t>(b) * heads + h) * head_dim;

    if (sequence_length <= 0) {
        for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
            output[query_base + d] = 0.0f;
        }
        return;
    }

    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        acc[d] = 0.0f;
    }
    if (threadIdx.x == 0) max_score = -3.402823e38f;
    __syncthreads();

    for (int token = 0; token < sequence_length; ++token) {
        const int logical_block = token / block_size;
        const int block_offset = token - logical_block * block_size;
        if (logical_block >= max_blocks) return;
        const int physical_block = static_cast<int>(block_table[logical_block]);
        if (physical_block < 0 || physical_block >= physical_blocks) return;
        const size_t kv_base = ((static_cast<size_t>(h) * physical_blocks + physical_block) * block_size + block_offset) * head_dim;

        float partial = 0.0f;
        for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
            partial += query[query_base + d] * key_blocks[kv_base + d];
        }
        reduce[threadIdx.x] = partial;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) reduce[threadIdx.x] += reduce[threadIdx.x + stride];
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            const float score = reduce[0] * rsqrtf(static_cast<float>(head_dim));
            max_score = fmaxf(max_score, score);
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) denom = 0.0f;
    __syncthreads();

    for (int token = 0; token < sequence_length; ++token) {
        const int logical_block = token / block_size;
        const int block_offset = token - logical_block * block_size;
        const int physical_block = static_cast<int>(block_table[logical_block]);
        const size_t kv_base = ((static_cast<size_t>(h) * physical_blocks + physical_block) * block_size + block_offset) * head_dim;

        float partial = 0.0f;
        for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
            partial += query[query_base + d] * key_blocks[kv_base + d];
        }
        reduce[threadIdx.x] = partial;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) reduce[threadIdx.x] += reduce[threadIdx.x + stride];
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            const float score = reduce[0] * rsqrtf(static_cast<float>(head_dim));
            weight = expf(score - max_score);
            denom += weight;
        }
        __syncthreads();
        for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
            acc[d] += weight * value_blocks[kv_base + d];
        }
        __syncthreads();
    }

    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        output[query_base + d] = acc[d] / denom;
    }
}

// Computes one routed-expert group with FP16 Tensor Core GEMMs:
//   gate = hidden @ gate_w.T ; up = hidden @ up_w.T
//   activation = silu(gate) * up   (FP16)
//   out = activation @ down_w.T
// Weights arrive as FP8 E4M3 + BF16 block scales and are dequantized to FP16
// tiles; hidden is float32 and is converted to FP16. Outputs stay FP32.
// Returns false (without launching GEMMs) if workspace allocation fails so the
// caller can fall back to the scalar kernels.
bool moe_expert_tensor_core_compute(
    const float* hidden_ptr,
    const uint8_t* gate_weight_ptr,
    const uint16_t* gate_scale_ptr,
    const uint8_t* up_weight_ptr,
    const uint16_t* up_scale_ptr,
    const uint8_t* down_weight_ptr,
    const uint16_t* down_scale_ptr,
    float* output_ptr,
    int batch,
    int hidden_size,
    int intermediate_size,
    cudaStream_t stream) {
    MoeTensorCoreWorkspace& ws = moe_tensor_core_workspace();
    if (!ws.ensure(static_cast<size_t>(batch), static_cast<size_t>(hidden_size), static_cast<size_t>(intermediate_size))) {
        return false;
    }

    const int threads = 256;
    const int input_scale_cols = hidden_size / kFp8BlockSize;
    const int down_scale_cols = intermediate_size / kFp8BlockSize;
    const int64_t hidden_elems = static_cast<int64_t>(batch) * hidden_size;
    const int64_t activation_elems = static_cast<int64_t>(batch) * intermediate_size;
    const int hidden_blocks = static_cast<int>(std::min<int64_t>((hidden_elems + threads - 1) / threads, 65535));
    const int activation_blocks = static_cast<int>(std::min<int64_t>((activation_elems + threads - 1) / threads, 65535));

    const int64_t gate_weight_elems = static_cast<int64_t>(intermediate_size) * hidden_size;
    const int64_t down_weight_elems = static_cast<int64_t>(hidden_size) * intermediate_size;
    const int gate_weight_blocks = static_cast<int>(std::min<int64_t>((gate_weight_elems + threads - 1) / threads, 65535));
    const int down_weight_blocks = static_cast<int>(std::min<int64_t>((down_weight_elems + threads - 1) / threads, 65535));

    float_rows_to_half_kernel<<<hidden_blocks, threads, 0, stream>>>(
        hidden_ptr, ws.d_hidden_half, static_cast<int>(hidden_elems));
    check_cuda(cudaGetLastError(), "moe_tc float_rows_to_half_kernel");
    fp8_weight_to_half_bf16_scale_kernel<<<gate_weight_blocks, threads, 0, stream>>>(
        gate_weight_ptr, gate_scale_ptr, ws.d_gate_w_half, intermediate_size, hidden_size, input_scale_cols);
    check_cuda(cudaGetLastError(), "moe_tc gate weight to half");
    fp8_weight_to_half_bf16_scale_kernel<<<gate_weight_blocks, threads, 0, stream>>>(
        up_weight_ptr, up_scale_ptr, ws.d_up_w_half, intermediate_size, hidden_size, input_scale_cols);
    check_cuda(cudaGetLastError(), "moe_tc up weight to half");
    fp8_weight_to_half_bf16_scale_kernel<<<down_weight_blocks, threads, 0, stream>>>(
        down_weight_ptr, down_scale_ptr, ws.d_down_w_half, hidden_size, intermediate_size, down_scale_cols);
    check_cuda(cudaGetLastError(), "moe_tc down weight to half");

    const __half* gate_w_half = ws.d_gate_w_half;
    const __half* up_w_half = ws.d_up_w_half;
    const __half* down_w_half = ws.d_down_w_half;

    check_cublas(cublasSetStream(ws.handle, stream), "moe_tc cublasSetStream");
    const float alpha = 1.0f;
    const float beta = 0.0f;
    // gate_float[batch, intermediate] = hidden_half[batch, hidden] @ gate_w_half[intermediate, hidden].T
    check_cublas(
        cublasGemmEx(ws.handle, CUBLAS_OP_T, CUBLAS_OP_N, intermediate_size, batch, hidden_size,
                     &alpha, gate_w_half, CUDA_R_16F, hidden_size, ws.d_hidden_half, CUDA_R_16F, hidden_size,
                     &beta, ws.d_gate_float, CUDA_R_32F, intermediate_size,
                     CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP),
        "moe_tc gate cublasGemmEx");
    check_cublas(
        cublasGemmEx(ws.handle, CUBLAS_OP_T, CUBLAS_OP_N, intermediate_size, batch, hidden_size,
                     &alpha, up_w_half, CUDA_R_16F, hidden_size, ws.d_hidden_half, CUDA_R_16F, hidden_size,
                     &beta, ws.d_up_float, CUDA_R_32F, intermediate_size,
                     CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP),
        "moe_tc up cublasGemmEx");

    silu_mul_float_to_half_kernel<<<activation_blocks, threads, 0, stream>>>(
        ws.d_gate_float, ws.d_up_float, ws.d_activation_half, activation_elems);
    check_cuda(cudaGetLastError(), "moe_tc silu_mul_float_to_half_kernel");

    // output[batch, hidden] = activation_half[batch, intermediate] @ down_w_half[hidden, intermediate].T
    check_cublas(
        cublasGemmEx(ws.handle, CUBLAS_OP_T, CUBLAS_OP_N, hidden_size, batch, intermediate_size,
                     &alpha, down_w_half, CUDA_R_16F, intermediate_size, ws.d_activation_half, CUDA_R_16F, intermediate_size,
                     &beta, output_ptr, CUDA_R_32F, hidden_size,
                     CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP),
        "moe_tc down cublasGemmEx");
    return true;
}

}  // namespace


torch::Tensor paged_attention_decode(
    torch::Tensor query,
    torch::Tensor block_table,
    torch::Tensor key_blocks,
    torch::Tensor value_blocks,
    int64_t sequence_length,
    int64_t block_size) {
    TORCH_CHECK(query.is_cuda(), "query must be CUDA");
    TORCH_CHECK(block_table.is_cuda(), "block_table must be CUDA");
    TORCH_CHECK(key_blocks.is_cuda(), "key_blocks must be CUDA");
    TORCH_CHECK(value_blocks.is_cuda(), "value_blocks must be CUDA");
    TORCH_CHECK(query.is_contiguous(), "query must be contiguous");
    TORCH_CHECK(block_table.is_contiguous(), "block_table must be contiguous");
    TORCH_CHECK(key_blocks.is_contiguous(), "key_blocks must be contiguous");
    TORCH_CHECK(value_blocks.is_contiguous(), "value_blocks must be contiguous");
    TORCH_CHECK(query.scalar_type() == at::kFloat, "query must be float32");
    TORCH_CHECK(key_blocks.scalar_type() == at::kFloat, "key_blocks must be float32");
    TORCH_CHECK(value_blocks.scalar_type() == at::kFloat, "value_blocks must be float32");
    TORCH_CHECK(block_table.scalar_type() == at::kLong, "block_table must be int64/torch.long");
    TORCH_CHECK(query.dim() == 4, "query must have shape [batch, heads, 1, head_dim]");
    TORCH_CHECK(key_blocks.dim() == 5, "key_blocks must have shape [batch, heads, blocks, block_size, head_dim]");
    TORCH_CHECK(value_blocks.dim() == 5, "value_blocks must have shape [batch, heads, blocks, block_size, head_dim]");
    TORCH_CHECK(block_table.dim() == 1, "block_table must be rank-1");
    TORCH_CHECK(query.size(2) == 1, "query seq_len must be 1 for paged decode");
    TORCH_CHECK(sequence_length > 0, "sequence_length must be positive");
    TORCH_CHECK(block_size > 0, "block_size must be positive");

    const auto batch64 = query.size(0);
    const auto heads64 = query.size(1);
    const auto head_dim64 = query.size(3);
    const auto physical_blocks64 = key_blocks.size(2);
    TORCH_CHECK(batch64 > 0 && batch64 <= INT_MAX, "batch must be positive and fit int32");
    TORCH_CHECK(heads64 > 0 && heads64 <= INT_MAX, "heads must be positive and fit int32");
    TORCH_CHECK(head_dim64 > 0 && head_dim64 <= 1024, "head_dim must be in [1, 1024]");
    TORCH_CHECK(physical_blocks64 > 0 && physical_blocks64 <= INT_MAX, "physical block count must be positive and fit int32");
    TORCH_CHECK(block_size <= INT_MAX, "block_size must fit int32");
    TORCH_CHECK(sequence_length <= INT_MAX, "sequence_length must fit int32");
    TORCH_CHECK(key_blocks.size(0) == batch64 && value_blocks.size(0) == batch64, "KV batch must match query");
    TORCH_CHECK(key_blocks.size(1) == heads64 && value_blocks.size(1) == heads64, "KV heads must match query");
    TORCH_CHECK(key_blocks.size(2) == value_blocks.size(2), "key/value physical block count must match");
    TORCH_CHECK(key_blocks.size(3) == block_size && value_blocks.size(3) == block_size, "KV block_size mismatch");
    TORCH_CHECK(key_blocks.size(4) == head_dim64 && value_blocks.size(4) == head_dim64, "KV head_dim must match query");
    const int64_t logical_blocks = (sequence_length + block_size - 1) / block_size;
    TORCH_CHECK(block_table.size(0) >= logical_blocks, "block_table does not cover sequence_length");

    c10::cuda::CUDAGuard device_guard(query.device());
    TORCH_CHECK(block_table.device() == query.device(), "block_table must be on same CUDA device as query");
    TORCH_CHECK(key_blocks.device() == query.device(), "key_blocks must be on same CUDA device as query");
    TORCH_CHECK(value_blocks.device() == query.device(), "value_blocks must be on same CUDA device as query");

    const int batch = static_cast<int>(batch64);
    const int heads = static_cast<int>(heads64);
    const int physical_blocks = static_cast<int>(physical_blocks64);
    const int block = static_cast<int>(block_size);
    const int head_dim = static_cast<int>(head_dim64);
    const int valid_tokens = static_cast<int>(sequence_length);
    auto output = torch::empty_like(query);
    auto stream = at::cuda::getCurrentCUDAStream(query.device().index()).stream();
    const int threads = 256;
    const size_t shared_bytes = static_cast<size_t>(threads + head_dim) * sizeof(float);
    paged_attention_decode_f32_kernel<<<batch * heads, threads, shared_bytes, stream>>>(
        query.data_ptr<float>(),
        block_table.data_ptr<int64_t>(),
        key_blocks.data_ptr<float>(),
        value_blocks.data_ptr<float>(),
        output.data_ptr<float>(),
        batch,
        heads,
        physical_blocks,
        block,
        head_dim,
        valid_tokens);
    check_cuda(cudaGetLastError(), "paged_attention_decode_f32_kernel");
    return output;
}


torch::Tensor paged_attention_decode_batched(
    torch::Tensor query,
    torch::Tensor block_tables,
    std::vector<torch::Tensor> key_blocks,
    std::vector<torch::Tensor> value_blocks,
    torch::Tensor sequence_lengths,
    int64_t block_size) {
    TORCH_CHECK(query.is_cuda(), "query must be CUDA");
    TORCH_CHECK(block_tables.is_cuda(), "block_tables must be CUDA");
    TORCH_CHECK(sequence_lengths.is_cuda(), "sequence_lengths must be CUDA");
    TORCH_CHECK(query.is_contiguous(), "query must be contiguous");
    TORCH_CHECK(block_tables.is_contiguous(), "block_tables must be contiguous");
    TORCH_CHECK(sequence_lengths.is_contiguous(), "sequence_lengths must be contiguous");
    TORCH_CHECK(query.scalar_type() == at::kFloat, "query must be float32");
    TORCH_CHECK(block_tables.scalar_type() == at::kLong, "block_tables must be int64/torch.long");
    TORCH_CHECK(sequence_lengths.scalar_type() == at::kLong, "sequence_lengths must be int64/torch.long");
    TORCH_CHECK(query.dim() == 4, "query must have shape [batch, heads, 1, head_dim]");
    TORCH_CHECK(block_tables.dim() == 2, "block_tables must have shape [batch, max_blocks]");
    TORCH_CHECK(sequence_lengths.dim() == 1, "sequence_lengths must be rank-1");
    TORCH_CHECK(query.size(2) == 1, "query seq_len must be 1 for paged decode");
    TORCH_CHECK(block_size > 0, "block_size must be positive");
    TORCH_CHECK(block_size <= INT_MAX, "block_size must fit int32");

    const auto batch64 = query.size(0);
    const auto heads64 = query.size(1);
    const auto head_dim64 = query.size(3);
    const auto max_blocks64 = block_tables.size(1);
    TORCH_CHECK(batch64 > 0 && batch64 <= INT_MAX, "batch must be positive and fit int32");
    TORCH_CHECK(heads64 > 0 && heads64 <= INT_MAX, "heads must be positive and fit int32");
    TORCH_CHECK(head_dim64 > 0 && head_dim64 <= 1024, "head_dim must be in [1, 1024]");
    TORCH_CHECK(max_blocks64 > 0 && max_blocks64 <= INT_MAX, "max_blocks must be positive and fit int32");
    TORCH_CHECK(block_tables.size(0) == batch64, "block_tables batch must match query");
    TORCH_CHECK(sequence_lengths.size(0) == batch64, "sequence_lengths batch must match query");
    TORCH_CHECK(static_cast<int64_t>(key_blocks.size()) == batch64, "key_blocks list length must match batch");
    TORCH_CHECK(static_cast<int64_t>(value_blocks.size()) == batch64, "value_blocks list length must match batch");

    c10::cuda::CUDAGuard device_guard(query.device());
    TORCH_CHECK(block_tables.device() == query.device(), "block_tables must be on same CUDA device as query");
    TORCH_CHECK(sequence_lengths.device() == query.device(), "sequence_lengths must be on same CUDA device as query");

    const size_t batch = static_cast<size_t>(batch64);
    std::vector<int64_t> key_ptr_host(batch);
    std::vector<int64_t> value_ptr_host(batch);
    std::vector<int64_t> physical_block_host(batch);
    for (size_t row = 0; row < batch; ++row) {
        const torch::Tensor& key = key_blocks[row];
        const torch::Tensor& value = value_blocks[row];
        TORCH_CHECK(key.is_cuda() && value.is_cuda(), "key/value blocks must be CUDA");
        TORCH_CHECK(key.device() == query.device() && value.device() == query.device(), "key/value blocks must be on same CUDA device as query");
        TORCH_CHECK(key.is_contiguous() && value.is_contiguous(), "key/value blocks must be contiguous");
        TORCH_CHECK(key.scalar_type() == at::kFloat && value.scalar_type() == at::kFloat, "key/value blocks must be float32");
        TORCH_CHECK(key.dim() == 5 && value.dim() == 5, "key/value blocks must have shape [1, heads, blocks, block_size, head_dim]");
        TORCH_CHECK(key.size(0) == 1 && value.size(0) == 1, "per-request KV pools must have batch dim 1");
        TORCH_CHECK(key.size(1) == heads64 && value.size(1) == heads64, "KV heads must match query");
        TORCH_CHECK(key.size(2) == value.size(2), "key/value physical block count must match");
        TORCH_CHECK(key.size(2) > 0 && key.size(2) <= INT_MAX, "physical block count must be positive and fit int32");
        TORCH_CHECK(key.size(3) == block_size && value.size(3) == block_size, "KV block_size mismatch");
        TORCH_CHECK(key.size(4) == head_dim64 && value.size(4) == head_dim64, "KV head_dim must match query");
        key_ptr_host[row] = reinterpret_cast<int64_t>(key.data_ptr<float>());
        value_ptr_host[row] = reinterpret_cast<int64_t>(value.data_ptr<float>());
        physical_block_host[row] = key.size(2);
    }

    auto pointer_options = query.options().dtype(at::kLong);
    auto key_ptrs = torch::empty({batch64}, pointer_options);
    auto value_ptrs = torch::empty({batch64}, pointer_options);
    auto physical_block_counts = torch::empty({batch64}, pointer_options);
    auto stream = at::cuda::getCurrentCUDAStream(query.device().index()).stream();
    const size_t pointer_bytes = batch * sizeof(int64_t);
    check_cuda(cudaMemcpyAsync(key_ptrs.data_ptr<int64_t>(), key_ptr_host.data(), pointer_bytes, cudaMemcpyHostToDevice, stream), "copy key block pointers");
    check_cuda(cudaMemcpyAsync(value_ptrs.data_ptr<int64_t>(), value_ptr_host.data(), pointer_bytes, cudaMemcpyHostToDevice, stream), "copy value block pointers");
    check_cuda(cudaMemcpyAsync(physical_block_counts.data_ptr<int64_t>(), physical_block_host.data(), pointer_bytes, cudaMemcpyHostToDevice, stream), "copy physical block counts");

    const int batch_int = static_cast<int>(batch64);
    const int heads = static_cast<int>(heads64);
    const int head_dim = static_cast<int>(head_dim64);
    const int max_blocks = static_cast<int>(max_blocks64);
    const int block = static_cast<int>(block_size);
    auto output = torch::empty_like(query);
    const int threads = 256;
    const size_t shared_bytes = static_cast<size_t>(threads + head_dim) * sizeof(float);
    paged_attention_decode_batched_f32_kernel<<<batch_int * heads, threads, shared_bytes, stream>>>(
        query.data_ptr<float>(),
        block_tables.data_ptr<int64_t>(),
        reinterpret_cast<const float* const*>(key_ptrs.data_ptr<int64_t>()),
        reinterpret_cast<const float* const*>(value_ptrs.data_ptr<int64_t>()),
        physical_block_counts.data_ptr<int64_t>(),
        sequence_lengths.data_ptr<int64_t>(),
        output.data_ptr<float>(),
        batch_int,
        heads,
        max_blocks,
        block,
        head_dim);
    check_cuda(cudaGetLastError(), "paged_attention_decode_batched_f32_kernel");
    return output;
}


torch::Tensor fp8_e4m3_bf16_linear(torch::Tensor input, torch::Tensor weight, torch::Tensor scale) {
    TORCH_CHECK(input.is_cuda(), "input must be CUDA");
    TORCH_CHECK(weight.is_cuda(), "weight must be CUDA");
    TORCH_CHECK(scale.is_cuda(), "scale must be CUDA");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
    TORCH_CHECK(scale.is_contiguous(), "scale must be contiguous");
    TORCH_CHECK(input.scalar_type() == at::kFloat, "input must be float32");
    TORCH_CHECK(weight.scalar_type() == at::kFloat8_e4m3fn, "weight must be torch.float8_e4m3fn");
    TORCH_CHECK(scale.scalar_type() == at::kBFloat16, "scale must be bfloat16");
    TORCH_CHECK(weight.dim() == 2, "weight must be rank-2");
    TORCH_CHECK(scale.dim() == 2, "scale must be rank-2");
    TORCH_CHECK(input.dim() >= 2, "input must be at least rank-2");

    const auto rows64 = weight.size(0);
    const auto cols64 = weight.size(1);
    TORCH_CHECK(input.size(-1) == cols64, "input last dim must match weight cols");
    TORCH_CHECK(rows64 > 0 && cols64 > 0, "weight shape must be positive");
    TORCH_CHECK(rows64 % kFp8BlockSize == 0, "weight rows must be divisible by 128 for the C++ FP8 op");
    TORCH_CHECK(cols64 % kFp8BlockSize == 0, "weight cols must be divisible by 128 for the C++ FP8 op");
    TORCH_CHECK(scale.size(0) == rows64 / kFp8BlockSize, "scale row blocks mismatch");
    TORCH_CHECK(scale.size(1) == cols64 / kFp8BlockSize, "scale col blocks mismatch");

    const int rows = static_cast<int>(rows64);
    const int cols = static_cast<int>(cols64);
    const int64_t batch64 = input.numel() / cols64;
    TORCH_CHECK(batch64 > 0 && batch64 <= INT_MAX, "batch too large");
    const int batch = static_cast<int>(batch64);

    c10::cuda::CUDAGuard device_guard(input.device());
    TORCH_CHECK(weight.device() == input.device(), "weight must be on same CUDA device as input");
    TORCH_CHECK(scale.device() == input.device(), "scale must be on same CUDA device as input");
    TORCH_CHECK(ensure_fp8_lut(), "failed to initialize FP8 LUT");

    auto flat_input = input.view({batch, cols});
    auto out = torch::empty({batch, rows}, input.options());
    auto stream = at::cuda::getCurrentCUDAStream(input.device().index()).stream();
    const auto* x_ptr = flat_input.data_ptr<float>();
    const auto* w_ptr = reinterpret_cast<const uint8_t*>(weight.data_ptr());
    const auto* s_ptr = reinterpret_cast<const uint16_t*>(scale.data_ptr<at::BFloat16>());
    auto* y_ptr = out.data_ptr<float>();
    const int scale_cols = cols / kFp8BlockSize;
    const int threads = 256;

    if (batch < kCublasBatchThreshold) {
        fp8_e4m3_bf16_matvec_kernel<<<dim3(rows, batch), threads, threads * sizeof(float), stream>>>(
            x_ptr, w_ptr, s_ptr, y_ptr, rows, cols, scale_cols);
        check_cuda(cudaGetLastError(), "fp8_e4m3_bf16_matvec_kernel");
    } else {
        Fp8Workspace& ws = workspace();
        TORCH_CHECK(ws.ensure(static_cast<size_t>(batch) * cols, static_cast<size_t>(rows) * cols), "failed to allocate FP8 workspace");
        const int64_t weight_elems = static_cast<int64_t>(rows) * cols;
        const int64_t x_elems = static_cast<int64_t>(batch) * cols;
        const int weight_blocks = static_cast<int>(std::min<int64_t>((weight_elems + threads - 1) / threads, 65535));
        const int x_blocks = static_cast<int>(std::min<int64_t>((x_elems + threads - 1) / threads, 65535));

        fp8_weight_to_half_bf16_scale_kernel<<<weight_blocks, threads, 0, stream>>>(
            w_ptr, s_ptr, ws.d_w_half, rows, cols, scale_cols);
        check_cuda(cudaGetLastError(), "fp8_weight_to_half_bf16_scale_kernel");
        float_rows_to_half_kernel<<<x_blocks, threads, 0, stream>>>(x_ptr, ws.d_x_half, static_cast<int>(x_elems));
        check_cuda(cudaGetLastError(), "float_rows_to_half_kernel");

        check_cublas(cublasSetStream(ws.handle, stream), "cublasSetStream");
        const float alpha = 1.0f;
        const float beta = 0.0f;
        check_cublas(
            cublasGemmEx(
                ws.handle,
                CUBLAS_OP_T,
                CUBLAS_OP_N,
                rows,
                batch,
                cols,
                &alpha,
                ws.d_w_half,
                CUDA_R_16F,
                cols,
                ws.d_x_half,
                CUDA_R_16F,
                cols,
                &beta,
                y_ptr,
                CUDA_R_32F,
                rows,
                CUBLAS_COMPUTE_32F,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP),
            "cublasGemmEx");
    }

    auto out_shape = input.sizes().vec();
    out_shape.back() = rows;
    return out.view(out_shape);
}


torch::Tensor fp8_e4m3_bf16_moe_expert(
    torch::Tensor hidden,
    torch::Tensor gate_weight,
    torch::Tensor gate_scale,
    torch::Tensor up_weight,
    torch::Tensor up_scale,
    torch::Tensor down_weight,
    torch::Tensor down_scale) {
    check_moe_expert_tensor_basics(hidden, "hidden", at::kFloat);
    check_moe_expert_tensor_basics(gate_weight, "gate_weight", at::kFloat8_e4m3fn);
    check_moe_expert_tensor_basics(gate_scale, "gate_scale", at::kBFloat16);
    check_moe_expert_tensor_basics(up_weight, "up_weight", at::kFloat8_e4m3fn);
    check_moe_expert_tensor_basics(up_scale, "up_scale", at::kBFloat16);
    check_moe_expert_tensor_basics(down_weight, "down_weight", at::kFloat8_e4m3fn);
    check_moe_expert_tensor_basics(down_scale, "down_scale", at::kBFloat16);

    const auto batch64 = hidden.size(0);
    const auto hidden64 = hidden.size(1);
    const auto intermediate64 = gate_weight.size(0);
    TORCH_CHECK(batch64 > 0 && batch64 <= INT_MAX, "batch must be positive and fit int32");
    TORCH_CHECK(hidden64 > 0 && hidden64 <= INT_MAX, "hidden size must be positive and fit int32");
    TORCH_CHECK(intermediate64 > 0 && intermediate64 <= INT_MAX, "intermediate size must be positive and fit int32");
    TORCH_CHECK(gate_weight.size(1) == hidden64, "gate_weight cols must match hidden size");
    TORCH_CHECK(up_weight.size(0) == intermediate64 && up_weight.size(1) == hidden64, "up_weight shape must match gate_weight");
    TORCH_CHECK(down_weight.size(0) == hidden64 && down_weight.size(1) == intermediate64, "down_weight shape must be [hidden, intermediate]");
    TORCH_CHECK(hidden64 % kFp8BlockSize == 0, "hidden size must be divisible by 128 for fused MoE expert op");
    TORCH_CHECK(intermediate64 % kFp8BlockSize == 0, "intermediate size must be divisible by 128 for fused MoE expert op");
    TORCH_CHECK(gate_scale.size(0) == intermediate64 / kFp8BlockSize, "gate_scale row blocks mismatch");
    TORCH_CHECK(gate_scale.size(1) == hidden64 / kFp8BlockSize, "gate_scale col blocks mismatch");
    TORCH_CHECK(up_scale.size(0) == intermediate64 / kFp8BlockSize, "up_scale row blocks mismatch");
    TORCH_CHECK(up_scale.size(1) == hidden64 / kFp8BlockSize, "up_scale col blocks mismatch");
    TORCH_CHECK(down_scale.size(0) == hidden64 / kFp8BlockSize, "down_scale row blocks mismatch");
    TORCH_CHECK(down_scale.size(1) == intermediate64 / kFp8BlockSize, "down_scale col blocks mismatch");

    c10::cuda::CUDAGuard device_guard(hidden.device());
    TORCH_CHECK(gate_weight.device() == hidden.device(), "gate_weight must be on same CUDA device as hidden");
    TORCH_CHECK(gate_scale.device() == hidden.device(), "gate_scale must be on same CUDA device as hidden");
    TORCH_CHECK(up_weight.device() == hidden.device(), "up_weight must be on same CUDA device as hidden");
    TORCH_CHECK(up_scale.device() == hidden.device(), "up_scale must be on same CUDA device as hidden");
    TORCH_CHECK(down_weight.device() == hidden.device(), "down_weight must be on same CUDA device as hidden");
    TORCH_CHECK(down_scale.device() == hidden.device(), "down_scale must be on same CUDA device as hidden");
    TORCH_CHECK(ensure_fp8_lut(), "failed to initialize FP8 LUT");

    const int batch = static_cast<int>(batch64);
    const int hidden_size = static_cast<int>(hidden64);
    const int intermediate_size = static_cast<int>(intermediate64);
    auto output = torch::empty({batch, hidden_size}, hidden.options());
    auto stream = at::cuda::getCurrentCUDAStream(hidden.device().index()).stream();
    const auto* hidden_ptr = hidden.data_ptr<float>();
    const auto* gate_weight_ptr = reinterpret_cast<const uint8_t*>(gate_weight.data_ptr());
    const auto* gate_scale_ptr = reinterpret_cast<const uint16_t*>(gate_scale.data_ptr<at::BFloat16>());
    const auto* up_weight_ptr = reinterpret_cast<const uint8_t*>(up_weight.data_ptr());
    const auto* up_scale_ptr = reinterpret_cast<const uint16_t*>(up_scale.data_ptr<at::BFloat16>());
    const auto* down_weight_ptr = reinterpret_cast<const uint8_t*>(down_weight.data_ptr());
    const auto* down_scale_ptr = reinterpret_cast<const uint16_t*>(down_scale.data_ptr<at::BFloat16>());

    const bool tensor_core_eligible = moe_tensor_core_enabled() &&
                                      device_has_fp16_tensor_core() &&
                                      batch >= moe_tensor_core_min_group_tokens() &&
                                      hidden_size % 8 == 0 &&
                                      intermediate_size % 8 == 0;
    if (tensor_core_eligible &&
        moe_expert_tensor_core_compute(
            hidden_ptr,
            gate_weight_ptr,
            gate_scale_ptr,
            up_weight_ptr,
            up_scale_ptr,
            down_weight_ptr,
            down_scale_ptr,
            output.data_ptr<float>(),
            batch,
            hidden_size,
            intermediate_size,
            stream)) {
        return output;
    }

    auto activation = torch::empty({batch, intermediate_size}, hidden.options());
    const int threads = 256;
    const int input_scale_cols = hidden_size / kFp8BlockSize;
    const int down_scale_cols = intermediate_size / kFp8BlockSize;

    fp8_moe_gate_up_silu_kernel<<<dim3(intermediate_size, batch), threads, 2 * threads * sizeof(float), stream>>>(
        hidden_ptr,
        gate_weight_ptr,
        gate_scale_ptr,
        up_weight_ptr,
        up_scale_ptr,
        activation.data_ptr<float>(),
        hidden_size,
        intermediate_size,
        input_scale_cols);
    check_cuda(cudaGetLastError(), "fp8_moe_gate_up_silu_kernel");

    fp8_moe_down_kernel<<<dim3(hidden_size, batch), threads, threads * sizeof(float), stream>>>(
        activation.data_ptr<float>(),
        down_weight_ptr,
        down_scale_ptr,
        output.data_ptr<float>(),
        hidden_size,
        intermediate_size,
        down_scale_cols);
    check_cuda(cudaGetLastError(), "fp8_moe_down_kernel");
    return output;
}

void moe_packed_score_scatter_add(
    torch::Tensor routed,
    torch::Tensor packed_output,
    torch::Tensor packed_tokens,
    torch::Tensor packed_scores) {
    TORCH_CHECK(routed.is_cuda(), "routed must be CUDA");
    TORCH_CHECK(packed_output.is_cuda(), "packed_output must be CUDA");
    TORCH_CHECK(packed_tokens.is_cuda(), "packed_tokens must be CUDA");
    TORCH_CHECK(packed_scores.is_cuda(), "packed_scores must be CUDA");
    TORCH_CHECK(routed.is_contiguous(), "routed must be contiguous");
    TORCH_CHECK(packed_output.is_contiguous(), "packed_output must be contiguous");
    TORCH_CHECK(packed_tokens.is_contiguous(), "packed_tokens must be contiguous");
    TORCH_CHECK(packed_scores.is_contiguous(), "packed_scores must be contiguous");
    TORCH_CHECK(routed.scalar_type() == at::kFloat, "routed must be float32");
    TORCH_CHECK(packed_output.scalar_type() == at::kFloat, "packed_output must be float32");
    TORCH_CHECK(packed_tokens.scalar_type() == at::kLong, "packed_tokens must be int64");
    TORCH_CHECK(packed_scores.scalar_type() == at::kFloat, "packed_scores must be float32");
    TORCH_CHECK(routed.dim() == 2, "routed must have shape [tokens, hidden]");
    TORCH_CHECK(packed_output.dim() == 2, "packed_output must have shape [assignments, hidden]");
    TORCH_CHECK(packed_tokens.dim() == 1, "packed_tokens must have shape [assignments]");
    TORCH_CHECK(packed_scores.dim() == 1, "packed_scores must have shape [assignments]");
    TORCH_CHECK(packed_tokens.size(0) == packed_output.size(0), "packed_tokens length must match packed_output assignments");
    TORCH_CHECK(packed_scores.size(0) == packed_output.size(0), "packed_scores length must match packed_output assignments");
    TORCH_CHECK(packed_output.size(1) == routed.size(1), "packed_output hidden size must match routed hidden size");
    TORCH_CHECK(routed.size(1) > 0 && routed.size(1) <= INT_MAX, "hidden size must be positive and fit int32");
    TORCH_CHECK(packed_output.size(0) >= 0, "assignment count must be non-negative");

    c10::cuda::CUDAGuard device_guard(routed.device());
    TORCH_CHECK(packed_output.device() == routed.device(), "packed_output must be on same CUDA device as routed");
    TORCH_CHECK(packed_tokens.device() == routed.device(), "packed_tokens must be on same CUDA device as routed");
    TORCH_CHECK(packed_scores.device() == routed.device(), "packed_scores must be on same CUDA device as routed");
    const int64_t assignments = packed_output.size(0);
    if (assignments == 0) return;
    const int hidden_size = static_cast<int>(routed.size(1));
    const int threads = 256;
    const int64_t total = assignments * static_cast<int64_t>(hidden_size);
    const int blocks = static_cast<int>(std::min<int64_t>((total + threads - 1) / threads, 65535));
    auto stream = at::cuda::getCurrentCUDAStream(routed.device().index()).stream();
    moe_packed_score_scatter_add_kernel<<<blocks, threads, 0, stream>>>(
        routed.data_ptr<float>(),
        packed_output.data_ptr<float>(),
        packed_tokens.data_ptr<int64_t>(),
        packed_scores.data_ptr<float>(),
        assignments,
        hidden_size);
    check_cuda(cudaGetLastError(), "moe_packed_score_scatter_add_kernel");
}


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
    int64_t token_count) {
    TORCH_CHECK(packed_hidden.is_cuda(), "packed_hidden must be CUDA");
    TORCH_CHECK(packed_tokens.is_cuda(), "packed_tokens must be CUDA");
    TORCH_CHECK(packed_scores.is_cuda(), "packed_scores must be CUDA");
    TORCH_CHECK(unique_experts.is_cuda(), "unique_experts must be CUDA");
    TORCH_CHECK(counts.is_cuda(), "counts must be CUDA");
    TORCH_CHECK(packed_hidden.is_contiguous(), "packed_hidden must be contiguous");
    TORCH_CHECK(packed_tokens.is_contiguous(), "packed_tokens must be contiguous");
    TORCH_CHECK(packed_scores.is_contiguous(), "packed_scores must be contiguous");
    TORCH_CHECK(unique_experts.is_contiguous(), "unique_experts must be contiguous");
    TORCH_CHECK(counts.is_contiguous(), "counts must be contiguous");
    TORCH_CHECK(packed_hidden.scalar_type() == at::kFloat, "packed_hidden must be float32");
    TORCH_CHECK(packed_tokens.scalar_type() == at::kLong, "packed_tokens must be int64");
    TORCH_CHECK(packed_scores.scalar_type() == at::kFloat, "packed_scores must be float32");
    TORCH_CHECK(unique_experts.scalar_type() == at::kLong, "unique_experts must be int64");
    TORCH_CHECK(counts.scalar_type() == at::kLong, "counts must be int64");
    TORCH_CHECK(packed_hidden.dim() == 2, "packed_hidden must have shape [assignments, hidden]");
    TORCH_CHECK(packed_tokens.dim() == 1, "packed_tokens must have shape [assignments]");
    TORCH_CHECK(packed_scores.dim() == 1, "packed_scores must have shape [assignments]");
    TORCH_CHECK(unique_experts.dim() == 1, "unique_experts must have shape [groups]");
    TORCH_CHECK(counts.dim() == 1, "counts must have shape [groups]");
    TORCH_CHECK(packed_tokens.size(0) == packed_hidden.size(0), "packed_tokens length must match packed_hidden assignments");
    TORCH_CHECK(packed_scores.size(0) == packed_hidden.size(0), "packed_scores length must match packed_hidden assignments");
    TORCH_CHECK(unique_experts.size(0) == counts.size(0), "unique_experts length must match counts");
    TORCH_CHECK(token_count >= 0, "token_count must be non-negative");
    TORCH_CHECK(packed_hidden.size(1) > 0 && packed_hidden.size(1) <= INT_MAX, "hidden size must be positive and fit int32");
    TORCH_CHECK(gate_weights.size() == gate_scales.size(), "gate weight/scale list length mismatch");
    TORCH_CHECK(gate_weights.size() == up_weights.size(), "gate/up weight list length mismatch");
    TORCH_CHECK(gate_weights.size() == up_scales.size(), "gate/up scale list length mismatch");
    TORCH_CHECK(gate_weights.size() == down_weights.size(), "gate/down weight list length mismatch");
    TORCH_CHECK(gate_weights.size() == down_scales.size(), "gate/down scale list length mismatch");
    TORCH_CHECK(!gate_weights.empty(), "expert tensor lists must be non-empty");

    c10::cuda::CUDAGuard device_guard(packed_hidden.device());
    TORCH_CHECK(packed_tokens.device() == packed_hidden.device(), "packed_tokens must be on same CUDA device as packed_hidden");
    TORCH_CHECK(packed_scores.device() == packed_hidden.device(), "packed_scores must be on same CUDA device as packed_hidden");
    TORCH_CHECK(unique_experts.device() == packed_hidden.device(), "unique_experts must be on same CUDA device as packed_hidden");
    TORCH_CHECK(counts.device() == packed_hidden.device(), "counts must be on same CUDA device as packed_hidden");
    TORCH_CHECK(ensure_fp8_lut(), "failed to initialize FP8 LUT");

    const auto hidden64 = packed_hidden.size(1);
    for (size_t index = 0; index < gate_weights.size(); ++index) {
        check_moe_expert_tensor_basics(gate_weights[index], "gate_weight", at::kFloat8_e4m3fn);
        check_moe_expert_tensor_basics(gate_scales[index], "gate_scale", at::kBFloat16);
        check_moe_expert_tensor_basics(up_weights[index], "up_weight", at::kFloat8_e4m3fn);
        check_moe_expert_tensor_basics(up_scales[index], "up_scale", at::kBFloat16);
        check_moe_expert_tensor_basics(down_weights[index], "down_weight", at::kFloat8_e4m3fn);
        check_moe_expert_tensor_basics(down_scales[index], "down_scale", at::kBFloat16);
        TORCH_CHECK(gate_weights[index].device() == packed_hidden.device(), "gate_weight must be on same CUDA device as packed_hidden");
        TORCH_CHECK(gate_scales[index].device() == packed_hidden.device(), "gate_scale must be on same CUDA device as packed_hidden");
        TORCH_CHECK(up_weights[index].device() == packed_hidden.device(), "up_weight must be on same CUDA device as packed_hidden");
        TORCH_CHECK(up_scales[index].device() == packed_hidden.device(), "up_scale must be on same CUDA device as packed_hidden");
        TORCH_CHECK(down_weights[index].device() == packed_hidden.device(), "down_weight must be on same CUDA device as packed_hidden");
        TORCH_CHECK(down_scales[index].device() == packed_hidden.device(), "down_scale must be on same CUDA device as packed_hidden");
        const auto intermediate64 = gate_weights[index].size(0);
        TORCH_CHECK(gate_weights[index].size(1) == hidden64, "gate_weight cols must match hidden size");
        TORCH_CHECK(up_weights[index].size(0) == intermediate64 && up_weights[index].size(1) == hidden64, "up_weight shape must match gate_weight");
        TORCH_CHECK(down_weights[index].size(0) == hidden64 && down_weights[index].size(1) == intermediate64, "down_weight shape must be [hidden, intermediate]");
        TORCH_CHECK(hidden64 % kFp8BlockSize == 0, "hidden size must be divisible by 128 for grouped MoE dispatch op");
        TORCH_CHECK(intermediate64 % kFp8BlockSize == 0, "intermediate size must be divisible by 128 for grouped MoE dispatch op");
        TORCH_CHECK(gate_scales[index].size(0) == intermediate64 / kFp8BlockSize, "gate_scale row blocks mismatch");
        TORCH_CHECK(gate_scales[index].size(1) == hidden64 / kFp8BlockSize, "gate_scale col blocks mismatch");
        TORCH_CHECK(up_scales[index].size(0) == intermediate64 / kFp8BlockSize, "up_scale row blocks mismatch");
        TORCH_CHECK(up_scales[index].size(1) == hidden64 / kFp8BlockSize, "up_scale col blocks mismatch");
        TORCH_CHECK(down_scales[index].size(0) == hidden64 / kFp8BlockSize, "down_scale row blocks mismatch");
        TORCH_CHECK(down_scales[index].size(1) == intermediate64 / kFp8BlockSize, "down_scale col blocks mismatch");
    }

    auto routed = torch::zeros({token_count, hidden64}, packed_hidden.options());
    const int64_t assignments = packed_hidden.size(0);
    const int64_t groups = unique_experts.size(0);
    if (assignments == 0 || groups == 0 || token_count == 0) return routed;

    auto packed_output = torch::empty({assignments, hidden64}, packed_hidden.options());
    auto unique_cpu = unique_experts.to(torch::kCPU);
    auto counts_cpu = counts.to(torch::kCPU);
    const auto* unique_data = unique_cpu.data_ptr<int64_t>();
    const auto* counts_data = counts_cpu.data_ptr<int64_t>();
    int64_t offset = 0;
    for (int64_t group = 0; group < groups; ++group) {
        const int64_t expert_id = unique_data[group];
        const int64_t count = counts_data[group];
        TORCH_CHECK(count >= 0, "counts must be non-negative");
        TORCH_CHECK(offset + count <= assignments, "counts exceed assignment count");
        if (count == 0) continue;
        const int64_t local_expert = expert_id - expert_start;
        TORCH_CHECK(local_expert >= 0 && static_cast<size_t>(local_expert) < gate_weights.size(), "unique expert out of local tensor-list range");
        auto hidden_chunk = packed_hidden.narrow(0, offset, count);
        auto token_output = fp8_e4m3_bf16_moe_expert(
            hidden_chunk,
            gate_weights[static_cast<size_t>(local_expert)],
            gate_scales[static_cast<size_t>(local_expert)],
            up_weights[static_cast<size_t>(local_expert)],
            up_scales[static_cast<size_t>(local_expert)],
            down_weights[static_cast<size_t>(local_expert)],
            down_scales[static_cast<size_t>(local_expert)]);
        packed_output.narrow(0, offset, count).copy_(token_output);
        offset += count;
    }
    TORCH_CHECK(offset == assignments, "counts total must match assignment count");
    moe_packed_score_scatter_add(routed, packed_output, packed_tokens, packed_scores);
    return routed;
}


torch::Tensor moe_grouped_dispatch_offsets_fp8_e4m3_bf16(
    torch::Tensor flat_hidden,
    torch::Tensor packed_tokens,
    torch::Tensor packed_scores,
    torch::Tensor counts,
    torch::Tensor offsets,
    int64_t expert_start,
    std::vector<torch::Tensor> gate_weights,
    std::vector<torch::Tensor> gate_scales,
    std::vector<torch::Tensor> up_weights,
    std::vector<torch::Tensor> up_scales,
    std::vector<torch::Tensor> down_weights,
    std::vector<torch::Tensor> down_scales,
    int64_t token_count) {
    TORCH_CHECK(flat_hidden.is_cuda(), "flat_hidden must be CUDA");
    TORCH_CHECK(packed_tokens.is_cuda(), "packed_tokens must be CUDA");
    TORCH_CHECK(packed_scores.is_cuda(), "packed_scores must be CUDA");
    TORCH_CHECK(counts.is_cuda(), "counts must be CUDA");
    TORCH_CHECK(offsets.is_cuda(), "offsets must be CUDA");
    TORCH_CHECK(flat_hidden.is_contiguous(), "flat_hidden must be contiguous");
    TORCH_CHECK(packed_tokens.is_contiguous(), "packed_tokens must be contiguous");
    TORCH_CHECK(packed_scores.is_contiguous(), "packed_scores must be contiguous");
    TORCH_CHECK(counts.is_contiguous(), "counts must be contiguous");
    TORCH_CHECK(offsets.is_contiguous(), "offsets must be contiguous");
    TORCH_CHECK(flat_hidden.scalar_type() == at::kFloat, "flat_hidden must be float32");
    TORCH_CHECK(packed_tokens.scalar_type() == at::kLong, "packed_tokens must be int64");
    TORCH_CHECK(packed_scores.scalar_type() == at::kFloat, "packed_scores must be float32");
    TORCH_CHECK(counts.scalar_type() == at::kLong, "counts must be int64");
    TORCH_CHECK(offsets.scalar_type() == at::kLong, "offsets must be int64");
    TORCH_CHECK(flat_hidden.dim() == 2, "flat_hidden must have shape [tokens, hidden]");
    TORCH_CHECK(packed_tokens.dim() == 1, "packed_tokens must have shape [capacity]");
    TORCH_CHECK(packed_scores.dim() == 1, "packed_scores must have shape [capacity]");
    TORCH_CHECK(counts.dim() == 1, "counts must have shape [local_experts]");
    TORCH_CHECK(offsets.dim() == 1, "offsets must have shape [local_experts]");
    TORCH_CHECK(packed_scores.size(0) == packed_tokens.size(0), "packed_scores length must match packed_tokens capacity");
    TORCH_CHECK(counts.size(0) == offsets.size(0), "counts length must match offsets length");
    TORCH_CHECK(token_count >= 0, "token_count must be non-negative");
    TORCH_CHECK(flat_hidden.size(1) > 0 && flat_hidden.size(1) <= INT_MAX, "hidden size must be positive and fit int32");
    TORCH_CHECK(gate_weights.size() == gate_scales.size(), "gate weight/scale list length mismatch");
    TORCH_CHECK(gate_weights.size() == up_weights.size(), "gate/up weight list length mismatch");
    TORCH_CHECK(gate_weights.size() == up_scales.size(), "gate/up scale list length mismatch");
    TORCH_CHECK(gate_weights.size() == down_weights.size(), "gate/down weight list length mismatch");
    TORCH_CHECK(gate_weights.size() == down_scales.size(), "gate/down scale list length mismatch");
    TORCH_CHECK(!gate_weights.empty(), "expert tensor lists must be non-empty");
    TORCH_CHECK(static_cast<int64_t>(gate_weights.size()) == counts.size(0), "expert tensor-list length must match counts length");

    c10::cuda::CUDAGuard device_guard(flat_hidden.device());
    TORCH_CHECK(packed_tokens.device() == flat_hidden.device(), "packed_tokens must be on same CUDA device as flat_hidden");
    TORCH_CHECK(packed_scores.device() == flat_hidden.device(), "packed_scores must be on same CUDA device as flat_hidden");
    TORCH_CHECK(counts.device() == flat_hidden.device(), "counts must be on same CUDA device as flat_hidden");
    TORCH_CHECK(offsets.device() == flat_hidden.device(), "offsets must be on same CUDA device as flat_hidden");
    TORCH_CHECK(ensure_fp8_lut(), "failed to initialize FP8 LUT");

    const auto hidden64 = flat_hidden.size(1);
    for (size_t index = 0; index < gate_weights.size(); ++index) {
        check_moe_expert_tensor_basics(gate_weights[index], "gate_weight", at::kFloat8_e4m3fn);
        check_moe_expert_tensor_basics(gate_scales[index], "gate_scale", at::kBFloat16);
        check_moe_expert_tensor_basics(up_weights[index], "up_weight", at::kFloat8_e4m3fn);
        check_moe_expert_tensor_basics(up_scales[index], "up_scale", at::kBFloat16);
        check_moe_expert_tensor_basics(down_weights[index], "down_weight", at::kFloat8_e4m3fn);
        check_moe_expert_tensor_basics(down_scales[index], "down_scale", at::kBFloat16);
        TORCH_CHECK(gate_weights[index].device() == flat_hidden.device(), "gate_weight must be on same CUDA device as flat_hidden");
        TORCH_CHECK(gate_scales[index].device() == flat_hidden.device(), "gate_scale must be on same CUDA device as flat_hidden");
        TORCH_CHECK(up_weights[index].device() == flat_hidden.device(), "up_weight must be on same CUDA device as flat_hidden");
        TORCH_CHECK(up_scales[index].device() == flat_hidden.device(), "up_scale must be on same CUDA device as flat_hidden");
        TORCH_CHECK(down_weights[index].device() == flat_hidden.device(), "down_weight must be on same CUDA device as flat_hidden");
        TORCH_CHECK(down_scales[index].device() == flat_hidden.device(), "down_scale must be on same CUDA device as flat_hidden");
        const auto intermediate64 = gate_weights[index].size(0);
        TORCH_CHECK(gate_weights[index].size(1) == hidden64, "gate_weight cols must match hidden size");
        TORCH_CHECK(up_weights[index].size(0) == intermediate64 && up_weights[index].size(1) == hidden64, "up_weight shape must match gate_weight");
        TORCH_CHECK(down_weights[index].size(0) == hidden64 && down_weights[index].size(1) == intermediate64, "down_weight shape must be [hidden, intermediate]");
        TORCH_CHECK(hidden64 % kFp8BlockSize == 0, "hidden size must be divisible by 128 for grouped MoE offset dispatch op");
        TORCH_CHECK(intermediate64 % kFp8BlockSize == 0, "intermediate size must be divisible by 128 for grouped MoE offset dispatch op");
        TORCH_CHECK(gate_scales[index].size(0) == intermediate64 / kFp8BlockSize, "gate_scale row blocks mismatch");
        TORCH_CHECK(gate_scales[index].size(1) == hidden64 / kFp8BlockSize, "gate_scale col blocks mismatch");
        TORCH_CHECK(up_scales[index].size(0) == intermediate64 / kFp8BlockSize, "up_scale row blocks mismatch");
        TORCH_CHECK(up_scales[index].size(1) == hidden64 / kFp8BlockSize, "up_scale col blocks mismatch");
        TORCH_CHECK(down_scales[index].size(0) == hidden64 / kFp8BlockSize, "down_scale row blocks mismatch");
        TORCH_CHECK(down_scales[index].size(1) == intermediate64 / kFp8BlockSize, "down_scale col blocks mismatch");
    }

    auto routed = torch::zeros({token_count, hidden64}, flat_hidden.options());
    const int64_t local_experts = counts.size(0);
    if (token_count == 0 || local_experts == 0) return routed;

    auto counts_cpu = counts.to(torch::kCPU);
    auto offsets_cpu = offsets.to(torch::kCPU);
    const auto* counts_data = counts_cpu.data_ptr<int64_t>();
    const auto* offsets_data = offsets_cpu.data_ptr<int64_t>();
    for (int64_t local = 0; local < local_experts; ++local) {
        const int64_t count = counts_data[local];
        const int64_t offset = offsets_data[local];
        TORCH_CHECK(count >= 0, "counts must be non-negative");
        TORCH_CHECK(offset >= 0, "offsets must be non-negative");
        TORCH_CHECK(offset + count <= packed_tokens.size(0), "offset/count exceed packed assignment capacity");
        if (count == 0) continue;
        auto token_slice = packed_tokens.narrow(0, offset, count);
        auto score_slice = packed_scores.narrow(0, offset, count);
        auto hidden_chunk = flat_hidden.index_select(0, token_slice);
        auto token_output = fp8_e4m3_bf16_moe_expert(
            hidden_chunk,
            gate_weights[static_cast<size_t>(local)],
            gate_scales[static_cast<size_t>(local)],
            up_weights[static_cast<size_t>(local)],
            up_scales[static_cast<size_t>(local)],
            down_weights[static_cast<size_t>(local)],
            down_scales[static_cast<size_t>(local)]);
        moe_packed_score_scatter_add(routed, token_output, token_slice, score_slice);
    }
    return routed;
}


torch::Tensor moe_grouped_dispatch_offsets_segmented_fp8_e4m3_bf16(
    torch::Tensor flat_hidden,
    torch::Tensor packed_tokens,
    torch::Tensor packed_scores,
    torch::Tensor counts,
    torch::Tensor offsets,
    int64_t expert_start,
    std::vector<torch::Tensor> gate_weights,
    std::vector<torch::Tensor> gate_scales,
    std::vector<torch::Tensor> up_weights,
    std::vector<torch::Tensor> up_scales,
    std::vector<torch::Tensor> down_weights,
    std::vector<torch::Tensor> down_scales,
    int64_t token_count) {
    (void)expert_start;
    TORCH_CHECK(flat_hidden.is_cuda(), "flat_hidden must be CUDA");
    TORCH_CHECK(packed_tokens.is_cuda(), "packed_tokens must be CUDA");
    TORCH_CHECK(packed_scores.is_cuda(), "packed_scores must be CUDA");
    TORCH_CHECK(counts.is_cuda(), "counts must be CUDA");
    TORCH_CHECK(offsets.is_cuda(), "offsets must be CUDA");
    TORCH_CHECK(flat_hidden.is_contiguous(), "flat_hidden must be contiguous");
    TORCH_CHECK(packed_tokens.is_contiguous(), "packed_tokens must be contiguous");
    TORCH_CHECK(packed_scores.is_contiguous(), "packed_scores must be contiguous");
    TORCH_CHECK(counts.is_contiguous(), "counts must be contiguous");
    TORCH_CHECK(offsets.is_contiguous(), "offsets must be contiguous");
    TORCH_CHECK(flat_hidden.scalar_type() == at::kFloat, "flat_hidden must be float32");
    TORCH_CHECK(packed_tokens.scalar_type() == at::kLong, "packed_tokens must be int64");
    TORCH_CHECK(packed_scores.scalar_type() == at::kFloat, "packed_scores must be float32");
    TORCH_CHECK(counts.scalar_type() == at::kLong, "counts must be int64");
    TORCH_CHECK(offsets.scalar_type() == at::kLong, "offsets must be int64");
    TORCH_CHECK(flat_hidden.dim() == 2, "flat_hidden must have shape [tokens, hidden]");
    TORCH_CHECK(packed_tokens.dim() == 1, "packed_tokens must have shape [capacity]");
    TORCH_CHECK(packed_scores.dim() == 1, "packed_scores must have shape [capacity]");
    TORCH_CHECK(counts.dim() == 1, "counts must have shape [local_experts]");
    TORCH_CHECK(offsets.dim() == 1, "offsets must have shape [local_experts]");
    TORCH_CHECK(packed_scores.size(0) == packed_tokens.size(0), "packed_scores length must match packed_tokens capacity");
    TORCH_CHECK(counts.size(0) == offsets.size(0), "counts length must match offsets length");
    TORCH_CHECK(token_count >= 0, "token_count must be non-negative");
    TORCH_CHECK(token_count <= flat_hidden.size(0), "token_count must not exceed flat_hidden rows");
    TORCH_CHECK(flat_hidden.size(1) > 0 && flat_hidden.size(1) <= INT_MAX, "hidden size must be positive and fit int32");
    TORCH_CHECK(gate_weights.size() == gate_scales.size(), "gate weight/scale list length mismatch");
    TORCH_CHECK(gate_weights.size() == up_weights.size(), "gate/up weight list length mismatch");
    TORCH_CHECK(gate_weights.size() == up_scales.size(), "gate/up scale list length mismatch");
    TORCH_CHECK(gate_weights.size() == down_weights.size(), "gate/down weight list length mismatch");
    TORCH_CHECK(gate_weights.size() == down_scales.size(), "gate/down scale list length mismatch");
    TORCH_CHECK(!gate_weights.empty(), "expert tensor lists must be non-empty");
    TORCH_CHECK(static_cast<int64_t>(gate_weights.size()) == counts.size(0), "expert tensor-list length must match counts length");

    c10::cuda::CUDAGuard device_guard(flat_hidden.device());
    TORCH_CHECK(packed_tokens.device() == flat_hidden.device(), "packed_tokens must be on same CUDA device as flat_hidden");
    TORCH_CHECK(packed_scores.device() == flat_hidden.device(), "packed_scores must be on same CUDA device as flat_hidden");
    TORCH_CHECK(counts.device() == flat_hidden.device(), "counts must be on same CUDA device as flat_hidden");
    TORCH_CHECK(offsets.device() == flat_hidden.device(), "offsets must be on same CUDA device as flat_hidden");
    TORCH_CHECK(ensure_fp8_lut(), "failed to initialize FP8 LUT");

    const auto hidden64 = flat_hidden.size(1);
    const int64_t local_experts64 = counts.size(0);
    TORCH_CHECK(local_experts64 > 0 && local_experts64 <= INT_MAX, "local expert count must be positive and fit int32");
    const auto intermediate64 = gate_weights[0].size(0);
    TORCH_CHECK(intermediate64 > 0 && intermediate64 <= INT_MAX, "intermediate size must be positive and fit int32");
    for (size_t index = 0; index < gate_weights.size(); ++index) {
        check_moe_expert_tensor_basics(gate_weights[index], "gate_weight", at::kFloat8_e4m3fn);
        check_moe_expert_tensor_basics(gate_scales[index], "gate_scale", at::kBFloat16);
        check_moe_expert_tensor_basics(up_weights[index], "up_weight", at::kFloat8_e4m3fn);
        check_moe_expert_tensor_basics(up_scales[index], "up_scale", at::kBFloat16);
        check_moe_expert_tensor_basics(down_weights[index], "down_weight", at::kFloat8_e4m3fn);
        check_moe_expert_tensor_basics(down_scales[index], "down_scale", at::kBFloat16);
        TORCH_CHECK(gate_weights[index].device() == flat_hidden.device(), "gate_weight must be on same CUDA device as flat_hidden");
        TORCH_CHECK(gate_scales[index].device() == flat_hidden.device(), "gate_scale must be on same CUDA device as flat_hidden");
        TORCH_CHECK(up_weights[index].device() == flat_hidden.device(), "up_weight must be on same CUDA device as flat_hidden");
        TORCH_CHECK(up_scales[index].device() == flat_hidden.device(), "up_scale must be on same CUDA device as flat_hidden");
        TORCH_CHECK(down_weights[index].device() == flat_hidden.device(), "down_weight must be on same CUDA device as flat_hidden");
        TORCH_CHECK(down_scales[index].device() == flat_hidden.device(), "down_scale must be on same CUDA device as flat_hidden");
        TORCH_CHECK(gate_weights[index].size(0) == intermediate64, "all gate_weight rows must match");
        TORCH_CHECK(gate_weights[index].size(1) == hidden64, "gate_weight cols must match hidden size");
        TORCH_CHECK(up_weights[index].size(0) == intermediate64 && up_weights[index].size(1) == hidden64, "up_weight shape must match gate_weight");
        TORCH_CHECK(down_weights[index].size(0) == hidden64 && down_weights[index].size(1) == intermediate64, "down_weight shape must be [hidden, intermediate]");
        TORCH_CHECK(hidden64 % kFp8BlockSize == 0, "hidden size must be divisible by 128 for segmented MoE offset dispatch op");
        TORCH_CHECK(intermediate64 % kFp8BlockSize == 0, "intermediate size must be divisible by 128 for segmented MoE offset dispatch op");
        TORCH_CHECK(gate_scales[index].size(0) == intermediate64 / kFp8BlockSize, "gate_scale row blocks mismatch");
        TORCH_CHECK(gate_scales[index].size(1) == hidden64 / kFp8BlockSize, "gate_scale col blocks mismatch");
        TORCH_CHECK(up_scales[index].size(0) == intermediate64 / kFp8BlockSize, "up_scale row blocks mismatch");
        TORCH_CHECK(up_scales[index].size(1) == hidden64 / kFp8BlockSize, "up_scale col blocks mismatch");
        TORCH_CHECK(down_scales[index].size(0) == hidden64 / kFp8BlockSize, "down_scale row blocks mismatch");
        TORCH_CHECK(down_scales[index].size(1) == intermediate64 / kFp8BlockSize, "down_scale col blocks mismatch");
    }

    auto routed = torch::zeros({token_count, hidden64}, flat_hidden.options());
    const int64_t capacity64 = packed_tokens.size(0);
    if (token_count == 0 || capacity64 == 0 || local_experts64 == 0) return routed;

    auto activation = torch::empty({capacity64, intermediate64}, flat_hidden.options());
    auto pointer_options = flat_hidden.options().dtype(at::kLong);
    auto gate_weight_ptrs = torch::empty({local_experts64}, pointer_options);
    auto gate_scale_ptrs = torch::empty({local_experts64}, pointer_options);
    auto up_weight_ptrs = torch::empty({local_experts64}, pointer_options);
    auto up_scale_ptrs = torch::empty({local_experts64}, pointer_options);
    auto down_weight_ptrs = torch::empty({local_experts64}, pointer_options);
    auto down_scale_ptrs = torch::empty({local_experts64}, pointer_options);

    const size_t local_experts = static_cast<size_t>(local_experts64);
    std::vector<int64_t> gate_weight_host(local_experts);
    std::vector<int64_t> gate_scale_host(local_experts);
    std::vector<int64_t> up_weight_host(local_experts);
    std::vector<int64_t> up_scale_host(local_experts);
    std::vector<int64_t> down_weight_host(local_experts);
    std::vector<int64_t> down_scale_host(local_experts);
    for (size_t index = 0; index < local_experts; ++index) {
        gate_weight_host[index] = reinterpret_cast<int64_t>(gate_weights[index].data_ptr());
        gate_scale_host[index] = reinterpret_cast<int64_t>(gate_scales[index].data_ptr<at::BFloat16>());
        up_weight_host[index] = reinterpret_cast<int64_t>(up_weights[index].data_ptr());
        up_scale_host[index] = reinterpret_cast<int64_t>(up_scales[index].data_ptr<at::BFloat16>());
        down_weight_host[index] = reinterpret_cast<int64_t>(down_weights[index].data_ptr());
        down_scale_host[index] = reinterpret_cast<int64_t>(down_scales[index].data_ptr<at::BFloat16>());
    }

    auto stream = at::cuda::getCurrentCUDAStream(flat_hidden.device().index()).stream();
    const size_t pointer_bytes = local_experts * sizeof(int64_t);
    check_cuda(cudaMemcpyAsync(gate_weight_ptrs.data_ptr<int64_t>(), gate_weight_host.data(), pointer_bytes, cudaMemcpyHostToDevice, stream), "copy gate_weight pointers");
    check_cuda(cudaMemcpyAsync(gate_scale_ptrs.data_ptr<int64_t>(), gate_scale_host.data(), pointer_bytes, cudaMemcpyHostToDevice, stream), "copy gate_scale pointers");
    check_cuda(cudaMemcpyAsync(up_weight_ptrs.data_ptr<int64_t>(), up_weight_host.data(), pointer_bytes, cudaMemcpyHostToDevice, stream), "copy up_weight pointers");
    check_cuda(cudaMemcpyAsync(up_scale_ptrs.data_ptr<int64_t>(), up_scale_host.data(), pointer_bytes, cudaMemcpyHostToDevice, stream), "copy up_scale pointers");
    check_cuda(cudaMemcpyAsync(down_weight_ptrs.data_ptr<int64_t>(), down_weight_host.data(), pointer_bytes, cudaMemcpyHostToDevice, stream), "copy down_weight pointers");
    check_cuda(cudaMemcpyAsync(down_scale_ptrs.data_ptr<int64_t>(), down_scale_host.data(), pointer_bytes, cudaMemcpyHostToDevice, stream), "copy down_scale pointers");

    const int hidden_size = static_cast<int>(hidden64);
    const int intermediate_size = static_cast<int>(intermediate64);
    const int input_scale_cols = hidden_size / kFp8BlockSize;
    const int down_scale_cols = intermediate_size / kFp8BlockSize;
    const int threads = 256;
    const int local_experts_int = static_cast<int>(local_experts64);
    fp8_moe_segmented_gate_up_silu_kernel<<<dim3(intermediate_size, local_experts_int), threads, 2 * threads * sizeof(float), stream>>>(
        flat_hidden.data_ptr<float>(),
        packed_tokens.data_ptr<int64_t>(),
        counts.data_ptr<int64_t>(),
        offsets.data_ptr<int64_t>(),
        reinterpret_cast<const uint8_t* const*>(gate_weight_ptrs.data_ptr<int64_t>()),
        reinterpret_cast<const uint16_t* const*>(gate_scale_ptrs.data_ptr<int64_t>()),
        reinterpret_cast<const uint8_t* const*>(up_weight_ptrs.data_ptr<int64_t>()),
        reinterpret_cast<const uint16_t* const*>(up_scale_ptrs.data_ptr<int64_t>()),
        activation.data_ptr<float>(),
        capacity64,
        token_count,
        hidden_size,
        intermediate_size,
        input_scale_cols);
    check_cuda(cudaGetLastError(), "fp8_moe_segmented_gate_up_silu_kernel");

    fp8_moe_segmented_down_scatter_kernel<<<dim3(hidden_size, local_experts_int), threads, threads * sizeof(float), stream>>>(
        routed.data_ptr<float>(),
        activation.data_ptr<float>(),
        packed_tokens.data_ptr<int64_t>(),
        packed_scores.data_ptr<float>(),
        counts.data_ptr<int64_t>(),
        offsets.data_ptr<int64_t>(),
        reinterpret_cast<const uint8_t* const*>(down_weight_ptrs.data_ptr<int64_t>()),
        reinterpret_cast<const uint16_t* const*>(down_scale_ptrs.data_ptr<int64_t>()),
        capacity64,
        token_count,
        hidden_size,
        intermediate_size,
        down_scale_cols);
    check_cuda(cudaGetLastError(), "fp8_moe_segmented_down_scatter_kernel");
    return routed;
}


torch::Tensor moe_grouped_dispatch_offsets_assignment_fp8_e4m3_bf16(
    torch::Tensor flat_hidden,
    torch::Tensor packed_tokens,
    torch::Tensor packed_scores,
    torch::Tensor counts,
    torch::Tensor offsets,
    torch::Tensor packed_local_experts,
    int64_t expert_start,
    std::vector<torch::Tensor> gate_weights,
    std::vector<torch::Tensor> gate_scales,
    std::vector<torch::Tensor> up_weights,
    std::vector<torch::Tensor> up_scales,
    std::vector<torch::Tensor> down_weights,
    std::vector<torch::Tensor> down_scales,
    int64_t token_count) {
    (void)expert_start;
    TORCH_CHECK(flat_hidden.is_cuda(), "flat_hidden must be CUDA");
    TORCH_CHECK(packed_tokens.is_cuda(), "packed_tokens must be CUDA");
    TORCH_CHECK(packed_scores.is_cuda(), "packed_scores must be CUDA");
    TORCH_CHECK(counts.is_cuda(), "counts must be CUDA");
    TORCH_CHECK(offsets.is_cuda(), "offsets must be CUDA");
    TORCH_CHECK(packed_local_experts.is_cuda(), "packed_local_experts must be CUDA");
    TORCH_CHECK(flat_hidden.is_contiguous(), "flat_hidden must be contiguous");
    TORCH_CHECK(packed_tokens.is_contiguous(), "packed_tokens must be contiguous");
    TORCH_CHECK(packed_scores.is_contiguous(), "packed_scores must be contiguous");
    TORCH_CHECK(counts.is_contiguous(), "counts must be contiguous");
    TORCH_CHECK(offsets.is_contiguous(), "offsets must be contiguous");
    TORCH_CHECK(packed_local_experts.is_contiguous(), "packed_local_experts must be contiguous");
    TORCH_CHECK(flat_hidden.scalar_type() == at::kFloat, "flat_hidden must be float32");
    TORCH_CHECK(packed_tokens.scalar_type() == at::kLong, "packed_tokens must be int64");
    TORCH_CHECK(packed_scores.scalar_type() == at::kFloat, "packed_scores must be float32");
    TORCH_CHECK(counts.scalar_type() == at::kLong, "counts must be int64");
    TORCH_CHECK(offsets.scalar_type() == at::kLong, "offsets must be int64");
    TORCH_CHECK(packed_local_experts.scalar_type() == at::kLong, "packed_local_experts must be int64");
    TORCH_CHECK(flat_hidden.dim() == 2, "flat_hidden must have shape [tokens, hidden]");
    TORCH_CHECK(packed_tokens.dim() == 1, "packed_tokens must have shape [capacity]");
    TORCH_CHECK(packed_scores.dim() == 1, "packed_scores must have shape [capacity]");
    TORCH_CHECK(packed_local_experts.dim() == 1, "packed_local_experts must have shape [capacity]");
    TORCH_CHECK(counts.dim() == 1, "counts must have shape [local_experts]");
    TORCH_CHECK(offsets.dim() == 1, "offsets must have shape [local_experts]");
    TORCH_CHECK(packed_scores.size(0) == packed_tokens.size(0), "packed_scores length must match packed_tokens capacity");
    TORCH_CHECK(packed_local_experts.size(0) == packed_tokens.size(0), "packed_local_experts length must match packed_tokens capacity");
    TORCH_CHECK(counts.size(0) == offsets.size(0), "counts length must match offsets length");
    TORCH_CHECK(token_count >= 0, "token_count must be non-negative");
    TORCH_CHECK(token_count <= flat_hidden.size(0), "token_count must not exceed flat_hidden rows");
    TORCH_CHECK(flat_hidden.size(1) > 0 && flat_hidden.size(1) <= INT_MAX, "hidden size must be positive and fit int32");
    TORCH_CHECK(gate_weights.size() == gate_scales.size(), "gate weight/scale list length mismatch");
    TORCH_CHECK(gate_weights.size() == up_weights.size(), "gate/up weight list length mismatch");
    TORCH_CHECK(gate_weights.size() == up_scales.size(), "gate/up scale list length mismatch");
    TORCH_CHECK(gate_weights.size() == down_weights.size(), "gate/down weight list length mismatch");
    TORCH_CHECK(gate_weights.size() == down_scales.size(), "gate/down scale list length mismatch");
    TORCH_CHECK(!gate_weights.empty(), "expert tensor lists must be non-empty");
    TORCH_CHECK(static_cast<int64_t>(gate_weights.size()) == counts.size(0), "expert tensor-list length must match counts length");

    c10::cuda::CUDAGuard device_guard(flat_hidden.device());
    TORCH_CHECK(packed_tokens.device() == flat_hidden.device(), "packed_tokens must be on same CUDA device as flat_hidden");
    TORCH_CHECK(packed_scores.device() == flat_hidden.device(), "packed_scores must be on same CUDA device as flat_hidden");
    TORCH_CHECK(counts.device() == flat_hidden.device(), "counts must be on same CUDA device as flat_hidden");
    TORCH_CHECK(offsets.device() == flat_hidden.device(), "offsets must be on same CUDA device as flat_hidden");
    TORCH_CHECK(packed_local_experts.device() == flat_hidden.device(), "packed_local_experts must be on same CUDA device as flat_hidden");
    TORCH_CHECK(ensure_fp8_lut(), "failed to initialize FP8 LUT");

    const auto hidden64 = flat_hidden.size(1);
    const int64_t local_experts64 = counts.size(0);
    TORCH_CHECK(local_experts64 > 0 && local_experts64 <= INT_MAX, "local expert count must be positive and fit int32");
    const auto intermediate64 = gate_weights[0].size(0);
    TORCH_CHECK(intermediate64 > 0 && intermediate64 <= INT_MAX, "intermediate size must be positive and fit int32");
    for (size_t index = 0; index < gate_weights.size(); ++index) {
        check_moe_expert_tensor_basics(gate_weights[index], "gate_weight", at::kFloat8_e4m3fn);
        check_moe_expert_tensor_basics(gate_scales[index], "gate_scale", at::kBFloat16);
        check_moe_expert_tensor_basics(up_weights[index], "up_weight", at::kFloat8_e4m3fn);
        check_moe_expert_tensor_basics(up_scales[index], "up_scale", at::kBFloat16);
        check_moe_expert_tensor_basics(down_weights[index], "down_weight", at::kFloat8_e4m3fn);
        check_moe_expert_tensor_basics(down_scales[index], "down_scale", at::kBFloat16);
        TORCH_CHECK(gate_weights[index].device() == flat_hidden.device(), "gate_weight must be on same CUDA device as flat_hidden");
        TORCH_CHECK(gate_scales[index].device() == flat_hidden.device(), "gate_scale must be on same CUDA device as flat_hidden");
        TORCH_CHECK(up_weights[index].device() == flat_hidden.device(), "up_weight must be on same CUDA device as flat_hidden");
        TORCH_CHECK(up_scales[index].device() == flat_hidden.device(), "up_scale must be on same CUDA device as flat_hidden");
        TORCH_CHECK(down_weights[index].device() == flat_hidden.device(), "down_weight must be on same CUDA device as flat_hidden");
        TORCH_CHECK(down_scales[index].device() == flat_hidden.device(), "down_scale must be on same CUDA device as flat_hidden");
        TORCH_CHECK(gate_weights[index].size(0) == intermediate64, "all gate_weight rows must match");
        TORCH_CHECK(gate_weights[index].size(1) == hidden64, "gate_weight cols must match hidden size");
        TORCH_CHECK(up_weights[index].size(0) == intermediate64 && up_weights[index].size(1) == hidden64, "up_weight shape must match gate_weight");
        TORCH_CHECK(down_weights[index].size(0) == hidden64 && down_weights[index].size(1) == intermediate64, "down_weight shape must be [hidden, intermediate]");
        TORCH_CHECK(hidden64 % kFp8BlockSize == 0, "hidden size must be divisible by 128 for assignment-parallel MoE offset dispatch op");
        TORCH_CHECK(intermediate64 % kFp8BlockSize == 0, "intermediate size must be divisible by 128 for assignment-parallel MoE offset dispatch op");
        TORCH_CHECK(gate_scales[index].size(0) == intermediate64 / kFp8BlockSize, "gate_scale row blocks mismatch");
        TORCH_CHECK(gate_scales[index].size(1) == hidden64 / kFp8BlockSize, "gate_scale col blocks mismatch");
        TORCH_CHECK(up_scales[index].size(0) == intermediate64 / kFp8BlockSize, "up_scale row blocks mismatch");
        TORCH_CHECK(up_scales[index].size(1) == hidden64 / kFp8BlockSize, "up_scale col blocks mismatch");
        TORCH_CHECK(down_scales[index].size(0) == hidden64 / kFp8BlockSize, "down_scale row blocks mismatch");
        TORCH_CHECK(down_scales[index].size(1) == intermediate64 / kFp8BlockSize, "down_scale col blocks mismatch");
    }

    auto routed = torch::zeros({token_count, hidden64}, flat_hidden.options());
    const int64_t capacity64 = packed_tokens.size(0);
    if (token_count == 0 || capacity64 == 0 || local_experts64 == 0) return routed;
    TORCH_CHECK(capacity64 <= INT_MAX, "packed assignment capacity must fit int32 grid dimension");

    auto activation = torch::empty({capacity64, intermediate64}, flat_hidden.options());
    auto pointer_options = flat_hidden.options().dtype(at::kLong);
    auto gate_weight_ptrs = torch::empty({local_experts64}, pointer_options);
    auto gate_scale_ptrs = torch::empty({local_experts64}, pointer_options);
    auto up_weight_ptrs = torch::empty({local_experts64}, pointer_options);
    auto up_scale_ptrs = torch::empty({local_experts64}, pointer_options);
    auto down_weight_ptrs = torch::empty({local_experts64}, pointer_options);
    auto down_scale_ptrs = torch::empty({local_experts64}, pointer_options);

    const size_t local_experts = static_cast<size_t>(local_experts64);
    std::vector<int64_t> gate_weight_host(local_experts);
    std::vector<int64_t> gate_scale_host(local_experts);
    std::vector<int64_t> up_weight_host(local_experts);
    std::vector<int64_t> up_scale_host(local_experts);
    std::vector<int64_t> down_weight_host(local_experts);
    std::vector<int64_t> down_scale_host(local_experts);
    for (size_t index = 0; index < local_experts; ++index) {
        gate_weight_host[index] = reinterpret_cast<int64_t>(gate_weights[index].data_ptr());
        gate_scale_host[index] = reinterpret_cast<int64_t>(gate_scales[index].data_ptr<at::BFloat16>());
        up_weight_host[index] = reinterpret_cast<int64_t>(up_weights[index].data_ptr());
        up_scale_host[index] = reinterpret_cast<int64_t>(up_scales[index].data_ptr<at::BFloat16>());
        down_weight_host[index] = reinterpret_cast<int64_t>(down_weights[index].data_ptr());
        down_scale_host[index] = reinterpret_cast<int64_t>(down_scales[index].data_ptr<at::BFloat16>());
    }

    auto stream = at::cuda::getCurrentCUDAStream(flat_hidden.device().index()).stream();
    const size_t pointer_bytes = local_experts * sizeof(int64_t);
    check_cuda(cudaMemcpyAsync(gate_weight_ptrs.data_ptr<int64_t>(), gate_weight_host.data(), pointer_bytes, cudaMemcpyHostToDevice, stream), "copy assignment gate_weight pointers");
    check_cuda(cudaMemcpyAsync(gate_scale_ptrs.data_ptr<int64_t>(), gate_scale_host.data(), pointer_bytes, cudaMemcpyHostToDevice, stream), "copy assignment gate_scale pointers");
    check_cuda(cudaMemcpyAsync(up_weight_ptrs.data_ptr<int64_t>(), up_weight_host.data(), pointer_bytes, cudaMemcpyHostToDevice, stream), "copy assignment up_weight pointers");
    check_cuda(cudaMemcpyAsync(up_scale_ptrs.data_ptr<int64_t>(), up_scale_host.data(), pointer_bytes, cudaMemcpyHostToDevice, stream), "copy assignment up_scale pointers");
    check_cuda(cudaMemcpyAsync(down_weight_ptrs.data_ptr<int64_t>(), down_weight_host.data(), pointer_bytes, cudaMemcpyHostToDevice, stream), "copy assignment down_weight pointers");
    check_cuda(cudaMemcpyAsync(down_scale_ptrs.data_ptr<int64_t>(), down_scale_host.data(), pointer_bytes, cudaMemcpyHostToDevice, stream), "copy assignment down_scale pointers");

    const int hidden_size = static_cast<int>(hidden64);
    const int intermediate_size = static_cast<int>(intermediate64);
    const int input_scale_cols = hidden_size / kFp8BlockSize;
    const int down_scale_cols = intermediate_size / kFp8BlockSize;
    const int threads = 256;
    const int local_experts_int = static_cast<int>(local_experts64);
    fp8_moe_assignment_gate_up_silu_kernel<<<dim3(intermediate_size, static_cast<unsigned int>(capacity64)), threads, 2 * threads * sizeof(float), stream>>>(
        flat_hidden.data_ptr<float>(),
        packed_tokens.data_ptr<int64_t>(),
        packed_local_experts.data_ptr<int64_t>(),
        reinterpret_cast<const uint8_t* const*>(gate_weight_ptrs.data_ptr<int64_t>()),
        reinterpret_cast<const uint16_t* const*>(gate_scale_ptrs.data_ptr<int64_t>()),
        reinterpret_cast<const uint8_t* const*>(up_weight_ptrs.data_ptr<int64_t>()),
        reinterpret_cast<const uint16_t* const*>(up_scale_ptrs.data_ptr<int64_t>()),
        activation.data_ptr<float>(),
        capacity64,
        token_count,
        local_experts_int,
        hidden_size,
        intermediate_size,
        input_scale_cols);
    check_cuda(cudaGetLastError(), "fp8_moe_assignment_gate_up_silu_kernel");

    fp8_moe_assignment_down_scatter_kernel<<<dim3(hidden_size, static_cast<unsigned int>(capacity64)), threads, threads * sizeof(float), stream>>>(
        routed.data_ptr<float>(),
        activation.data_ptr<float>(),
        packed_tokens.data_ptr<int64_t>(),
        packed_scores.data_ptr<float>(),
        packed_local_experts.data_ptr<int64_t>(),
        reinterpret_cast<const uint8_t* const*>(down_weight_ptrs.data_ptr<int64_t>()),
        reinterpret_cast<const uint16_t* const*>(down_scale_ptrs.data_ptr<int64_t>()),
        capacity64,
        token_count,
        local_experts_int,
        hidden_size,
        intermediate_size,
        down_scale_cols);
    check_cuda(cudaGetLastError(), "fp8_moe_assignment_down_scatter_kernel");
    return routed;
}


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> moe_packed_local_assignment_offsets(
    torch::Tensor indices,
    torch::Tensor scores,
    int64_t expert_start,
    int64_t expert_end) {
    TORCH_CHECK(indices.is_cuda(), "indices must be CUDA");
    TORCH_CHECK(scores.is_cuda(), "scores must be CUDA");
    TORCH_CHECK(indices.is_contiguous(), "indices must be contiguous");
    TORCH_CHECK(scores.is_contiguous(), "scores must be contiguous");
    TORCH_CHECK(indices.scalar_type() == at::kLong, "indices must be int64");
    TORCH_CHECK(scores.scalar_type() == at::kFloat, "scores must be float32");
    TORCH_CHECK(indices.dim() == 2, "indices must have shape [tokens, top_k]");
    TORCH_CHECK(scores.dim() == 2, "scores must have shape [tokens, top_k]");
    TORCH_CHECK(indices.sizes() == scores.sizes(), "indices and scores shapes must match");
    TORCH_CHECK(expert_end >= expert_start, "expert_end must be >= expert_start");

    c10::cuda::CUDAGuard device_guard(indices.device());
    TORCH_CHECK(scores.device() == indices.device(), "scores must be on same CUDA device as indices");

    const auto tokens64 = indices.size(0);
    const auto top_k64 = indices.size(1);
    TORCH_CHECK(tokens64 >= 0, "token count must be non-negative");
    TORCH_CHECK(top_k64 > 0 && top_k64 <= INT_MAX, "top_k must be positive and fit int32");
    TORCH_CHECK(tokens64 == 0 || tokens64 <= (std::numeric_limits<int64_t>::max() / top_k64), "assignment count overflow");
    const int top_k = static_cast<int>(top_k64);
    const int64_t total_assignments = tokens64 * top_k64;
    const int64_t local_experts64 = expert_end - expert_start;
    TORCH_CHECK(local_experts64 >= 0 && local_experts64 <= INT_MAX, "local expert count must fit int32");

    auto long_options = indices.options().dtype(at::kLong);
    auto score_options = scores.options();
    auto packed_tokens = torch::empty({total_assignments}, long_options);
    auto packed_scores = torch::empty({total_assignments}, score_options);
    auto unique_experts = torch::empty({local_experts64}, long_options);
    auto counts = torch::zeros({local_experts64}, long_options);
    auto offsets = torch::zeros({local_experts64}, long_options);
    if (local_experts64 == 0) {
        return std::make_tuple(packed_tokens, packed_scores, unique_experts, counts, offsets);
    }

    auto stream = at::cuda::getCurrentCUDAStream(indices.device().index()).stream();
    const int threads = 256;
    const int blocks = static_cast<int>(std::min<int64_t>((std::max<int64_t>(total_assignments, 1) + threads - 1) / threads, 65535));
    if (total_assignments > 0) {
        moe_assignment_count_kernel<<<blocks, threads, 0, stream>>>(
            indices.data_ptr<int64_t>(),
            counts.data_ptr<int64_t>(),
            total_assignments,
            top_k,
            expert_start,
            expert_end);
        check_cuda(cudaGetLastError(), "moe_assignment_count_kernel");
    }
    moe_assignment_offsets_kernel<<<1, 1, 0, stream>>>(
        counts.data_ptr<int64_t>(),
        offsets.data_ptr<int64_t>(),
        unique_experts.data_ptr<int64_t>(),
        local_experts64,
        expert_start);
    check_cuda(cudaGetLastError(), "moe_assignment_offsets_kernel");
    if (total_assignments > 0) {
        auto cursors = offsets.clone();
        moe_assignment_fill_kernel<<<blocks, threads, 0, stream>>>(
            indices.data_ptr<int64_t>(),
            scores.data_ptr<float>(),
            cursors.data_ptr<int64_t>(),
            packed_tokens.data_ptr<int64_t>(),
            packed_scores.data_ptr<float>(),
            total_assignments,
            top_k,
            expert_start,
            expert_end);
        check_cuda(cudaGetLastError(), "moe_assignment_fill_kernel");
    }
    return std::make_tuple(packed_tokens, packed_scores, unique_experts, counts, offsets);
}


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> moe_packed_local_assignment_offsets_with_experts(
    torch::Tensor indices,
    torch::Tensor scores,
    int64_t expert_start,
    int64_t expert_end) {
    TORCH_CHECK(indices.is_cuda(), "indices must be CUDA");
    TORCH_CHECK(scores.is_cuda(), "scores must be CUDA");
    TORCH_CHECK(indices.is_contiguous(), "indices must be contiguous");
    TORCH_CHECK(scores.is_contiguous(), "scores must be contiguous");
    TORCH_CHECK(indices.scalar_type() == at::kLong, "indices must be int64");
    TORCH_CHECK(scores.scalar_type() == at::kFloat, "scores must be float32");
    TORCH_CHECK(indices.dim() == 2, "indices must have shape [tokens, top_k]");
    TORCH_CHECK(scores.dim() == 2, "scores must have shape [tokens, top_k]");
    TORCH_CHECK(indices.sizes() == scores.sizes(), "indices and scores shapes must match");
    TORCH_CHECK(expert_end >= expert_start, "expert_end must be >= expert_start");

    c10::cuda::CUDAGuard device_guard(indices.device());
    TORCH_CHECK(scores.device() == indices.device(), "scores must be on same CUDA device as indices");

    const auto tokens64 = indices.size(0);
    const auto top_k64 = indices.size(1);
    TORCH_CHECK(tokens64 >= 0, "token count must be non-negative");
    TORCH_CHECK(top_k64 > 0 && top_k64 <= INT_MAX, "top_k must be positive and fit int32");
    TORCH_CHECK(tokens64 == 0 || tokens64 <= (std::numeric_limits<int64_t>::max() / top_k64), "assignment count overflow");
    const int top_k = static_cast<int>(top_k64);
    const int64_t total_assignments = tokens64 * top_k64;
    const int64_t local_experts64 = expert_end - expert_start;
    TORCH_CHECK(local_experts64 >= 0 && local_experts64 <= INT_MAX, "local expert count must fit int32");

    auto long_options = indices.options().dtype(at::kLong);
    auto score_options = scores.options();
    auto packed_tokens = torch::empty({total_assignments}, long_options);
    auto packed_scores = torch::empty({total_assignments}, score_options);
    auto packed_local_experts = torch::full({total_assignments}, -1, long_options);
    auto unique_experts = torch::empty({local_experts64}, long_options);
    auto counts = torch::zeros({local_experts64}, long_options);
    auto offsets = torch::zeros({local_experts64}, long_options);
    if (local_experts64 == 0) {
        return std::make_tuple(packed_tokens, packed_scores, unique_experts, counts, offsets, packed_local_experts);
    }

    auto stream = at::cuda::getCurrentCUDAStream(indices.device().index()).stream();
    const int threads = 256;
    const int blocks = static_cast<int>(std::min<int64_t>((std::max<int64_t>(total_assignments, 1) + threads - 1) / threads, 65535));
    if (total_assignments > 0) {
        moe_assignment_count_kernel<<<blocks, threads, 0, stream>>>(
            indices.data_ptr<int64_t>(),
            counts.data_ptr<int64_t>(),
            total_assignments,
            top_k,
            expert_start,
            expert_end);
        check_cuda(cudaGetLastError(), "moe_assignment_count_kernel");
    }
    moe_assignment_offsets_kernel<<<1, 1, 0, stream>>>(
        counts.data_ptr<int64_t>(),
        offsets.data_ptr<int64_t>(),
        unique_experts.data_ptr<int64_t>(),
        local_experts64,
        expert_start);
    check_cuda(cudaGetLastError(), "moe_assignment_offsets_kernel");
    if (total_assignments > 0) {
        auto cursors = offsets.clone();
        moe_assignment_fill_with_experts_kernel<<<blocks, threads, 0, stream>>>(
            indices.data_ptr<int64_t>(),
            scores.data_ptr<float>(),
            cursors.data_ptr<int64_t>(),
            packed_tokens.data_ptr<int64_t>(),
            packed_scores.data_ptr<float>(),
            packed_local_experts.data_ptr<int64_t>(),
            total_assignments,
            top_k,
            expert_start,
            expert_end);
        check_cuda(cudaGetLastError(), "moe_assignment_fill_with_experts_kernel");
    }
    return std::make_tuple(packed_tokens, packed_scores, unique_experts, counts, offsets, packed_local_experts);
}


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> moe_packed_local_assignments(
    torch::Tensor indices,
    torch::Tensor scores,
    int64_t expert_start,
    int64_t expert_end) {
    TORCH_CHECK(indices.is_cuda(), "indices must be CUDA");
    TORCH_CHECK(scores.is_cuda(), "scores must be CUDA");
    TORCH_CHECK(indices.is_contiguous(), "indices must be contiguous");
    TORCH_CHECK(scores.is_contiguous(), "scores must be contiguous");
    TORCH_CHECK(indices.scalar_type() == at::kLong, "indices must be int64");
    TORCH_CHECK(scores.scalar_type() == at::kFloat, "scores must be float32");
    TORCH_CHECK(indices.dim() == 2, "indices must have shape [tokens, top_k]");
    TORCH_CHECK(scores.dim() == 2, "scores must have shape [tokens, top_k]");
    TORCH_CHECK(indices.sizes() == scores.sizes(), "indices and scores shapes must match");
    TORCH_CHECK(expert_end >= expert_start, "expert_end must be >= expert_start");

    c10::cuda::CUDAGuard device_guard(indices.device());
    TORCH_CHECK(scores.device() == indices.device(), "scores must be on same CUDA device as indices");

    const auto tokens64 = indices.size(0);
    const auto top_k64 = indices.size(1);
    TORCH_CHECK(tokens64 >= 0, "token count must be non-negative");
    TORCH_CHECK(top_k64 > 0 && top_k64 <= INT_MAX, "top_k must be positive and fit int32");
    TORCH_CHECK(tokens64 == 0 || tokens64 <= (std::numeric_limits<int64_t>::max() / top_k64), "assignment count overflow");
    const int top_k = static_cast<int>(top_k64);
    const int64_t total_assignments = tokens64 * top_k64;
    const int64_t local_experts64 = expert_end - expert_start;
    TORCH_CHECK(local_experts64 >= 0 && local_experts64 <= INT_MAX, "local expert count must fit int32");

    auto long_options = indices.options().dtype(at::kLong);
    auto score_options = scores.options();
    if (local_experts64 == 0 || total_assignments == 0) {
        auto packed_tokens = torch::empty({0}, long_options);
        auto packed_scores = torch::empty({0}, score_options);
        auto unique_experts = torch::empty({0}, long_options);
        auto counts = torch::empty({0}, long_options);
        return std::make_tuple(packed_tokens, packed_scores, unique_experts, counts);
    }

    auto counts_full = torch::zeros({local_experts64}, long_options);
    auto stream = at::cuda::getCurrentCUDAStream(indices.device().index()).stream();
    const int threads = 256;
    const int blocks = static_cast<int>(std::min<int64_t>((total_assignments + threads - 1) / threads, 65535));
    moe_assignment_count_kernel<<<blocks, threads, 0, stream>>>(
        indices.data_ptr<int64_t>(),
        counts_full.data_ptr<int64_t>(),
        total_assignments,
        top_k,
        expert_start,
        expert_end);
    check_cuda(cudaGetLastError(), "moe_assignment_count_kernel");

    auto counts_cpu = counts_full.to(torch::kCPU);
    const auto* counts_data = counts_cpu.data_ptr<int64_t>();
    std::vector<int64_t> offsets(static_cast<size_t>(local_experts64), 0);
    std::vector<int64_t> unique_experts_host;
    std::vector<int64_t> counts_host;
    unique_experts_host.reserve(static_cast<size_t>(local_experts64));
    counts_host.reserve(static_cast<size_t>(local_experts64));
    int64_t local_assignment_count = 0;
    for (int64_t local = 0; local < local_experts64; ++local) {
        const int64_t count = counts_data[local];
        offsets[static_cast<size_t>(local)] = local_assignment_count;
        if (count > 0) {
            unique_experts_host.push_back(expert_start + local);
            counts_host.push_back(count);
            local_assignment_count += count;
        }
    }

    auto unique_experts_cpu = torch::empty({static_cast<int64_t>(unique_experts_host.size())}, torch::TensorOptions().dtype(at::kLong).device(torch::kCPU));
    auto counts_compact_cpu = torch::empty({static_cast<int64_t>(counts_host.size())}, torch::TensorOptions().dtype(at::kLong).device(torch::kCPU));
    if (!unique_experts_host.empty()) {
        std::memcpy(unique_experts_cpu.data_ptr<int64_t>(), unique_experts_host.data(), unique_experts_host.size() * sizeof(int64_t));
        std::memcpy(counts_compact_cpu.data_ptr<int64_t>(), counts_host.data(), counts_host.size() * sizeof(int64_t));
    }
    auto unique_experts = unique_experts_cpu.to(long_options.device(indices.device()));
    auto counts = counts_compact_cpu.to(long_options.device(indices.device()));

    if (local_assignment_count == 0) {
        auto packed_tokens = torch::empty({0}, long_options);
        auto packed_scores = torch::empty({0}, score_options);
        return std::make_tuple(packed_tokens, packed_scores, unique_experts, counts);
    }

    auto offsets_cpu = torch::empty({local_experts64}, torch::TensorOptions().dtype(at::kLong).device(torch::kCPU));
    std::memcpy(offsets_cpu.data_ptr<int64_t>(), offsets.data(), offsets.size() * sizeof(int64_t));
    auto offsets_full = offsets_cpu.to(long_options.device(indices.device()));
    auto cursors = offsets_full.clone();
    auto packed_tokens = torch::empty({local_assignment_count}, long_options);
    auto packed_scores = torch::empty({local_assignment_count}, score_options);

    moe_assignment_fill_kernel<<<blocks, threads, 0, stream>>>(
        indices.data_ptr<int64_t>(),
        scores.data_ptr<float>(),
        cursors.data_ptr<int64_t>(),
        packed_tokens.data_ptr<int64_t>(),
        packed_scores.data_ptr<float>(),
        total_assignments,
        top_k,
        expert_start,
        expert_end);
    check_cuda(cudaGetLastError(), "moe_assignment_fill_kernel");
    return std::make_tuple(packed_tokens, packed_scores, unique_experts, counts);
}


std::tuple<torch::Tensor, torch::Tensor> linear_attention_recurrent_core(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor g,
    torch::Tensor beta,
    torch::Tensor initial_state) {
    TORCH_CHECK(query.is_cuda(), "query must be CUDA");
    TORCH_CHECK(key.is_cuda(), "key must be CUDA");
    TORCH_CHECK(value.is_cuda(), "value must be CUDA");
    TORCH_CHECK(g.is_cuda(), "g must be CUDA");
    TORCH_CHECK(beta.is_cuda(), "beta must be CUDA");
    TORCH_CHECK(initial_state.is_cuda(), "initial_state must be CUDA");
    TORCH_CHECK(query.is_contiguous(), "query must be contiguous");
    TORCH_CHECK(key.is_contiguous(), "key must be contiguous");
    TORCH_CHECK(value.is_contiguous(), "value must be contiguous");
    TORCH_CHECK(g.is_contiguous(), "g must be contiguous");
    TORCH_CHECK(beta.is_contiguous(), "beta must be contiguous");
    TORCH_CHECK(initial_state.is_contiguous(), "initial_state must be contiguous");
    TORCH_CHECK(query.scalar_type() == at::kFloat, "query must be float32");
    TORCH_CHECK(key.scalar_type() == at::kFloat, "key must be float32");
    TORCH_CHECK(value.scalar_type() == at::kFloat, "value must be float32");
    TORCH_CHECK(g.scalar_type() == at::kFloat, "g must be float32");
    TORCH_CHECK(beta.scalar_type() == at::kFloat, "beta must be float32");
    TORCH_CHECK(initial_state.scalar_type() == at::kFloat, "initial_state must be float32");
    TORCH_CHECK(query.dim() == 4, "query must have shape [batch, heads, seq_len, key_dim]");
    TORCH_CHECK(key.dim() == 4, "key must have shape [batch, heads, seq_len, key_dim]");
    TORCH_CHECK(value.dim() == 4, "value must have shape [batch, heads, seq_len, value_dim]");
    TORCH_CHECK(g.dim() == 3, "g must have shape [batch, heads, seq_len]");
    TORCH_CHECK(beta.dim() == 3, "beta must have shape [batch, heads, seq_len]");
    TORCH_CHECK(initial_state.dim() == 4, "initial_state must have shape [batch, heads, key_dim, value_dim]");

    const auto batch64 = query.size(0);
    const auto heads64 = query.size(1);
    const auto seq64 = query.size(2);
    const auto key_dim64 = query.size(3);
    const auto value_dim64 = value.size(3);
    TORCH_CHECK(batch64 > 0 && batch64 <= INT_MAX, "batch must be positive and fit int32");
    TORCH_CHECK(heads64 > 0 && heads64 <= INT_MAX, "heads must be positive and fit int32");
    TORCH_CHECK(seq64 > 0 && seq64 <= INT_MAX, "seq_len must be positive and fit int32");
    TORCH_CHECK(key_dim64 > 0 && key_dim64 <= INT_MAX, "key_dim must be positive and fit int32");
    TORCH_CHECK(value_dim64 > 0 && value_dim64 <= INT_MAX, "value_dim must be positive and fit int32");
    TORCH_CHECK(key.sizes() == query.sizes(), "key shape must match query shape");
    TORCH_CHECK(value.size(0) == batch64 && value.size(1) == heads64 && value.size(2) == seq64, "value batch/head/seq must match query");
    TORCH_CHECK(g.size(0) == batch64 && g.size(1) == heads64 && g.size(2) == seq64, "g shape must match [batch, heads, seq_len]");
    TORCH_CHECK(beta.size(0) == batch64 && beta.size(1) == heads64 && beta.size(2) == seq64, "beta shape must match [batch, heads, seq_len]");
    TORCH_CHECK(initial_state.size(0) == batch64 && initial_state.size(1) == heads64, "initial_state batch/head must match query");
    TORCH_CHECK(initial_state.size(2) == key_dim64 && initial_state.size(3) == value_dim64, "initial_state key/value dims must match query/value");

    c10::cuda::CUDAGuard device_guard(query.device());
    TORCH_CHECK(key.device() == query.device(), "key must be on same CUDA device as query");
    TORCH_CHECK(value.device() == query.device(), "value must be on same CUDA device as query");
    TORCH_CHECK(g.device() == query.device(), "g must be on same CUDA device as query");
    TORCH_CHECK(beta.device() == query.device(), "beta must be on same CUDA device as query");
    TORCH_CHECK(initial_state.device() == query.device(), "initial_state must be on same CUDA device as query");

    const int batch = static_cast<int>(batch64);
    const int heads = static_cast<int>(heads64);
    const int seq_len = static_cast<int>(seq64);
    const int key_dim = static_cast<int>(key_dim64);
    const int value_dim = static_cast<int>(value_dim64);
    auto output = torch::empty({batch, heads, seq_len, value_dim}, query.options());
    auto final_state = torch::empty({batch, heads, key_dim, value_dim}, initial_state.options());
    auto stream = at::cuda::getCurrentCUDAStream(query.device().index()).stream();
    const int threads = 256;
    const dim3 grid(batch * heads, value_dim);

    linear_attention_recurrent_core_kernel<<<grid, threads, threads * sizeof(float), stream>>>(
        query.data_ptr<float>(),
        key.data_ptr<float>(),
        value.data_ptr<float>(),
        g.data_ptr<float>(),
        beta.data_ptr<float>(),
        initial_state.data_ptr<float>(),
        output.data_ptr<float>(),
        final_state.data_ptr<float>(),
        heads,
        seq_len,
        key_dim,
        value_dim);
    check_cuda(cudaGetLastError(), "linear_attention_recurrent_core_kernel");
    return std::make_tuple(output, final_state);
}
