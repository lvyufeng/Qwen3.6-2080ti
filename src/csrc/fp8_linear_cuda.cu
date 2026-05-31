#include "fp8_linear.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <cstdint>
#include <stdexcept>

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

}  // namespace


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
    auto activation = torch::empty({batch, intermediate_size}, hidden.options());
    auto output = torch::empty({batch, hidden_size}, hidden.options());
    auto stream = at::cuda::getCurrentCUDAStream(hidden.device().index()).stream();
    const int threads = 256;
    const int input_scale_cols = hidden_size / kFp8BlockSize;
    const int down_scale_cols = intermediate_size / kFp8BlockSize;

    fp8_moe_gate_up_silu_kernel<<<dim3(intermediate_size, batch), threads, 2 * threads * sizeof(float), stream>>>(
        hidden.data_ptr<float>(),
        reinterpret_cast<const uint8_t*>(gate_weight.data_ptr()),
        reinterpret_cast<const uint16_t*>(gate_scale.data_ptr<at::BFloat16>()),
        reinterpret_cast<const uint8_t*>(up_weight.data_ptr()),
        reinterpret_cast<const uint16_t*>(up_scale.data_ptr<at::BFloat16>()),
        activation.data_ptr<float>(),
        hidden_size,
        intermediate_size,
        input_scale_cols);
    check_cuda(cudaGetLastError(), "fp8_moe_gate_up_silu_kernel");

    fp8_moe_down_kernel<<<dim3(hidden_size, batch), threads, threads * sizeof(float), stream>>>(
        activation.data_ptr<float>(),
        reinterpret_cast<const uint8_t*>(down_weight.data_ptr()),
        reinterpret_cast<const uint16_t*>(down_scale.data_ptr<at::BFloat16>()),
        output.data_ptr<float>(),
        hidden_size,
        intermediate_size,
        down_scale_cols);
    check_cuda(cudaGetLastError(), "fp8_moe_down_kernel");
    return output;
}
