#include "conv.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdexcept>

#include <cutlass/cutlass.h>
#include "cutlass/conv/kernel/default_conv2d_fprop.h"
#include "cutlass/conv/device/implicit_gemm_convolution.h"

namespace {
template <uint8_t CONV_IM2COL_TILE_K, uint8_t LOG_CONV_IM2COL_TILE_K>
inline void launch_conv_for_in_c(uint32_t in_c,
                                 const half* input,
                                 half* output,
                                 size_t in_h,
                                 size_t in_w,
                                 size_t out_c,
                                 const half* weights,
                                 const half* bias,
                                 const dim3& block,
                                 const dim3& grid,
                                 const size_t& conv_shared_mem) {
    switch (in_c) {
#define CASE(IC)                                                                                  \
        case IC:                                                                                  \
            conv_relu_nhwc_oihw_cuda<CONV_IM2COL_TILE_K, LOG_CONV_IM2COL_TILE_K, IC>              \
                <<<grid, block, conv_shared_mem>>>(input, output, in_h, in_w, out_c, weights,     \
                                                   bias);                                         \
            break;
        FOR_EACH_IN_C(CASE)
#undef CASE
        default:
            throw std::runtime_error("Invalid in_c " + std::to_string(in_c));
    }
}
} // namespace

__device__ static inline void LOAD_TILE(half* dst_a, half* dst_b, const half* weights,
                               uint32_t linear_tid, uint32_t tile_a_elems, uint32_t tile_b_elems, uint32_t block_threads,
                               uint32_t CONV_IM2COL_TILE_K, uint32_t LOG_CONV_IM2COL_TILE_K, uint32_t log_block_dim_x, uint32_t in_h, uint32_t in_w, uint32_t in_c, uint32_t out_c,
                               uint32_t total_k, const half* input, uint32_t block_h0, uint32_t block_w0, uint32_t block_o0, uint32_t k_base_val) {
    const uint32_t max_tile = max(tile_a_elems, tile_b_elems);
    for (uint32_t idx = linear_tid; idx < max_tile; idx += block_threads) {
        const uint32_t k_col = idx & (CONV_IM2COL_TILE_K - 1);
        const uint32_t k_idx = k_base_val + k_col;
        half val_a = 0;
        half val_b = 0;
        if (k_idx < total_k) {
            const uint32_t hw_idx = idx >> LOG_CONV_IM2COL_TILE_K;
            const uint32_t local_x = hw_idx & (blockDim.x - 1);
            const uint32_t local_y = hw_idx >> log_block_dim_x;
            const uint32_t h_out = block_h0 + local_x;
            const uint32_t w_out = block_w0 + local_y;
            if (h_out < in_h && w_out < in_w && idx < tile_a_elems) {
                const uint32_t spatial = k_idx / in_c;
                const uint32_t ic = k_idx - spatial * in_c;
                const uint32_t fh = spatial / 3;
                const uint32_t fw = spatial - fh * 3;
                const uint32_t ih = h_out + fh - 1;
                const uint32_t iw = w_out + fw - 1;
                if (ih < in_h && iw < in_w) {
                    const uint32_t input_idx = ((ih * in_w + iw) * in_c) + ic;
                    val_a = input[input_idx];
                }
            }

            if (k_idx < total_k && idx < tile_b_elems) {
                const uint32_t o_out_local = idx >> LOG_CONV_IM2COL_TILE_K;
                const uint32_t o_out = block_o0 + o_out_local;
                if (o_out < out_c) {
                    val_b = weights[o_out * total_k + k_idx];
                }
            }
        }
        if (idx < tile_a_elems) {
            dst_a[idx] = val_a;
        }
        if (idx < tile_b_elems) {
            dst_b[idx] = val_b;
        }
    }
}
                                        
template <uint8_t CONV_IM2COL_TILE_K, uint8_t LOG_CONV_IM2COL_TILE_K, uint32_t IN_C>
__global__ void conv_relu_nhwc_oihw_cuda(const half* input,
                                         half* output,
                                         size_t in_h,
                                         size_t in_w,
                                         size_t out_c,
                                         const half* weights,
                                         const half* bias) {
    extern __shared__ half smem[];

    const uint32_t block_threads = blockDim.x * blockDim.y * blockDim.z;
    const uint32_t linear_tid = (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
    const uint32_t local_hw = threadIdx.y * blockDim.x + threadIdx.x;

    const uint32_t block_h0 = blockIdx.x * blockDim.x;
    const uint32_t block_w0 = blockIdx.y * blockDim.y;
    const uint32_t block_o0 = blockIdx.z * blockDim.z;

    const uint32_t h = block_h0 + threadIdx.x;
    const uint32_t w = block_w0 + threadIdx.y;
    const uint32_t o = block_o0 + threadIdx.z;

    const uint32_t total_k = IN_C * 9;
    const uint32_t tile_a_elems = blockDim.x * blockDim.y * CONV_IM2COL_TILE_K;
    const uint32_t tile_b_elems = blockDim.z * CONV_IM2COL_TILE_K;

    const uint32_t log_block_dim_x = __ffs(blockDim.x) - 1;

    half acc = (o < out_c) ? bias[o] : __float2half(0.0f);

    half* curr_a = smem;
    half* curr_b = smem + tile_a_elems;
    half* next_a = smem + tile_a_elems + tile_b_elems;
    half* next_b = smem + tile_a_elems + tile_b_elems + tile_a_elems;

    uint32_t k_base = 0;

    LOAD_TILE(curr_a, curr_b, weights, linear_tid, tile_a_elems, tile_b_elems,
              block_threads, CONV_IM2COL_TILE_K, LOG_CONV_IM2COL_TILE_K, log_block_dim_x, in_h, in_w, IN_C, out_c,
              total_k, input, block_h0, block_w0, block_o0, k_base);
    __syncthreads();

    k_base += CONV_IM2COL_TILE_K;

    while (true) {
        const half* a_row = curr_a + local_hw * CONV_IM2COL_TILE_K;
        const half* b_col = curr_b + threadIdx.z * CONV_IM2COL_TILE_K;
        for (uint32_t kk = 0; kk < CONV_IM2COL_TILE_K; ++kk)
            acc += a_row[kk] * b_col[kk];

        if (k_base >= total_k) {
            break;
        }

        LOAD_TILE(next_a, next_b, weights, linear_tid, tile_a_elems, tile_b_elems,
                  block_threads, CONV_IM2COL_TILE_K, LOG_CONV_IM2COL_TILE_K, log_block_dim_x, in_h, in_w, IN_C, out_c,
                  total_k, input, block_h0, block_w0, block_o0, k_base);
        __syncthreads();

        half* temp_a = curr_a;
        half* temp_b = curr_b;
        curr_a = next_a;
        curr_b = next_b;
        next_a = temp_a;
        next_b = temp_b;

        k_base += CONV_IM2COL_TILE_K;
    }

    if (h < in_h && w < in_w && o < out_c) {
        const uint32_t out_idx = ((h * in_w + w) * out_c) + o;
        output[out_idx] = __hmax(acc, 0);
    }
}

void gpu_conv(const half* input,
                             half* output,
                             size_t in_h,
                             size_t in_w,
                             size_t in_c,
                             size_t out_c,
                             const half* weights,
                             const half* bias,
                             const dim3& block,
                             const dim3& grid,
                             const int& conv_im2col_tile_ks,
                             const size_t& conv_shared_mem,
                             bool cutlass_conv) {
    if (cutlass_conv) {
        using ElementA           = cutlass::half_t;
        using ElementB           = cutlass::half_t;
        using ElementC           = float;
        using ElementAccumulator = float;
        using ElementCompute     = float;
        using Conv2dFpropKernel = typename cutlass::conv::kernel::DefaultConv2dFprop<
            ElementA, cutlass::layout::TensorNHWC,
            ElementB, cutlass::layout::TensorNHWC,
            ElementC, cutlass::layout::TensorNHWC,
            ElementAccumulator,
            cutlass::arch::OpClassTensorOp,
            cutlass::arch::Sm80,
            cutlass::gemm::GemmShape<128, 128, 64>,
            cutlass::gemm::GemmShape<64, 64, 64>,
            cutlass::gemm::GemmShape<16, 8, 16>,
            cutlass::epilogue::thread::LinearCombination<
            ElementC,
            128 / cutlass::sizeof_bits<ElementC>::value,
            ElementAccumulator,
            ElementCompute
            >,
            cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
            3,
            cutlass::arch::OpMultiplyAdd,
            cutlass::conv::IteratorAlgorithm::kAnalytic
        >::Kernel;

        using Conv2dFprop = cutlass::conv::device::ImplicitGemmConvolution<Conv2dFpropKernel>;
        Conv2dFprop implicit_gemm_op;
    } else {
        switch (conv_im2col_tile_ks) {
            case 4:
                launch_conv_for_in_c<4, 2>(in_c, input, output, in_h, in_w, out_c, weights, bias,
                                           block, grid, conv_shared_mem);
                break;
            case 8:
                launch_conv_for_in_c<8, 3>(in_c, input, output, in_h, in_w, out_c, weights, bias,
                                            block, grid, conv_shared_mem);
                break;
            case 16:
                launch_conv_for_in_c<16, 4>(in_c, input, output, in_h, in_w, out_c, weights, bias,
                                             block, grid, conv_shared_mem);
                break;
            case 32:
                launch_conv_for_in_c<32, 5>(in_c, input, output, in_h, in_w, out_c, weights, bias,
                                             block, grid, conv_shared_mem);
                break;
            case 64:
                launch_conv_for_in_c<64, 6>(in_c, input, output, in_h, in_w, out_c, weights, bias,
                                             block, grid, conv_shared_mem);
                break;
            default:
                throw std::runtime_error("Invalid conv_im2col_tile_ks");
        }
    }
}


__global__ void max_pool_nhwc_cuda(const half* input,
                               half* output,
                               size_t in_h,
                               size_t in_w,
                               size_t in_c) {
    const size_t out_h = in_h / 2;
    const size_t out_w = in_w / 2;

    const int h = blockIdx.x * blockDim.x + threadIdx.x;
    const int w = blockIdx.y * blockDim.y + threadIdx.y;
    const int c = blockIdx.z * blockDim.z + threadIdx.z;

    if (h >= out_h || w >= out_w || c >= in_c)
        return;

    const int ih = h * 2;
    const int iw = w * 2;

    const half* base_in = input + (static_cast<size_t>(ih) * in_w + iw) * in_c + c;
    const size_t row_stride = static_cast<size_t>(in_w) * in_c;
    const size_t col_stride = in_c;

    half max_val = base_in[0];
    max_val = __hmax(max_val, base_in[col_stride]);
    max_val = __hmax(max_val, base_in[row_stride]);
    max_val = __hmax(max_val, base_in[row_stride + col_stride]);

    const size_t out_idx = (static_cast<size_t>(h) * out_w + w) * in_c + c;
    output[out_idx] = max_val;
}

__global__ void avg_pool_nhwc_cuda(const half* input,
                               half* output,
                               size_t in_h,
                               size_t in_w,
                               size_t in_c) {

    const size_t out_h = in_h / 2;
    const size_t out_w = in_w / 2;

    const int h = blockIdx.x * blockDim.x + threadIdx.x;
    const int w = blockIdx.y * blockDim.y + threadIdx.y;
    const int c = blockIdx.z * blockDim.z + threadIdx.z;

    if (h >= out_h || w >= out_w || c >= in_c)
        return;

    const int ih = h * 2;
    const int iw = w * 2;

    const half* base_in = input + (static_cast<size_t>(ih) * in_w + iw) * in_c + c;
    const size_t row_stride = static_cast<size_t>(in_w) * in_c;
    const size_t col_stride = in_c;

    half avg_val = base_in[0];
    avg_val = __hadd(avg_val, base_in[col_stride]);
    avg_val = __hadd(avg_val, base_in[row_stride]);
    avg_val = __hadd(avg_val, base_in[row_stride + col_stride]);

    const size_t out_idx = (static_cast<size_t>(h) * out_w + w) * in_c + c;
    output[out_idx] = __hdiv(avg_val, __float2half(4.0f));
}

__global__ void nn_upsample_nhwc_cuda(const half* input,
                                  half* output,
                                  size_t in_h,
                                  size_t in_w,
                                  size_t in_c) {
    const size_t out_h = in_h * 2;
    const size_t out_w = in_w * 2;

    const int h = blockIdx.x * blockDim.x + threadIdx.x;
    const int w = blockIdx.y * blockDim.y + threadIdx.y;
    const int c = blockIdx.z * blockDim.z + threadIdx.z;

    if (h >= out_h || w >= out_w || c >= in_c)
        return;

    const size_t in_idx = (static_cast<size_t>(h / 2) * in_w + (w / 2)) * in_c + c;
    const size_t out_idx = (static_cast<size_t>(h) * out_w + w) * in_c + c;
    output[out_idx] = input[in_idx];
}
