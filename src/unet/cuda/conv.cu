#include "conv.h"

#include <cuda_fp16.h>
#include <cstdint>

// NOTE: REQUIRED THAT BLOCK.DIM.Z = BLOCK.DIM.X * BLOCK.DIM.Y
__global__ void conv_relu_nchw_oihw_cuda(const half* input,
                                         half* output,
                                         size_t in_h,
                                         size_t in_w,
                                         size_t in_c,
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

    const uint32_t total_k = in_c * 9;
    const uint32_t tile_a_elems = blockDim.x * blockDim.y * CONV_IM2COL_TILE_K;
    const uint32_t tile_b_elems = blockDim.z * CONV_IM2COL_TILE_K;

    half* tile_a_curr = smem;
    half* tile_b_curr = tile_a_curr + tile_a_elems;
    half* tile_a_next = tile_b_curr + tile_b_elems;
    half* tile_b_next = tile_a_next + tile_a_elems;

    half acc = bias[o];

#define LOAD_TILE(dst_a, dst_b, k_base_val)                                \
        for (uint32_t idx = linear_tid; idx < tile_a_elems; idx += block_threads) { \
            const uint32_t k_col = idx & (CONV_IM2COL_TILE_K - 1);          \
            const uint32_t k_idx = (k_base_val) + k_col;                    \
                                                                            \
            half val_a = 0;                                                 \
            half val_b = 0;                                                 \
            if (k_idx < total_k) {                                          \
                const uint32_t hw_idx = idx >> LOG_CONV_IM2COL_TILE_K;      \
                const uint32_t o_out = block_o0 + hw_idx;                  \
                                                                            \
                const uint32_t local_x = hw_idx % blockDim.x;               \
                const uint32_t local_y = hw_idx / blockDim.x;               \
                const uint32_t h_out = block_h0 + local_x;                 \
                const uint32_t w_out = block_w0 + local_y;                 \
                                                                            \
                if (h_out < in_h && w_out < in_w) {                         \
                    const uint32_t ic = k_idx / 9;                          \
                    const uint32_t fh = (k_idx % 9) / 3;                     \
                    const uint32_t fw = k_idx % 3;                          \
                                                                            \
                    const uint32_t ih = h_out + fh - 1;                     \
                    const uint32_t iw = w_out + fw - 1;                     \
                    if (ih < in_h && iw < in_w) {                           \
                        const uint32_t input_idx = (ic * in_h + ih) * in_w + iw; \
                        val_a = input[input_idx];                           \
                    }                                                       \
                }                                                           \
                                                                            \
                if (o_out < out_c) {                                        \
                    val_b = weights[o_out * total_k + k_idx];               \
                }                                                           \
            }                                                               \
                                                                            \
            (dst_a)[idx] = val_a;                                           \
            (dst_b)[idx] = val_b;                                           \
        }

    half* curr_a = tile_a_curr;
    half* curr_b = tile_b_curr;
    half* next_a = tile_a_next;
    half* next_b = tile_b_next;

    uint32_t next_k_base = CONV_IM2COL_TILE_K;
    bool has_next = next_k_base < total_k;

    LOAD_TILE(curr_a, curr_b, 0);
    if (has_next) {
        LOAD_TILE(next_a, next_b, next_k_base);
    }
    __syncthreads();

    while (true) {
        const half* a_row = curr_a + local_hw * CONV_IM2COL_TILE_K;
        const half* b_col = curr_b + threadIdx.z * CONV_IM2COL_TILE_K;
        for (uint32_t kk = 0; kk < CONV_IM2COL_TILE_K; ++kk)
            acc += a_row[kk] * b_col[kk];


        if (!has_next) {
            break;
        }

        uint32_t current_k_base = next_k_base;
        next_k_base = current_k_base + CONV_IM2COL_TILE_K;

        half* temp_a = curr_a;
        half* temp_b = curr_b;
        curr_a = next_a;
        curr_b = next_b;
        next_a = temp_a;
        next_b = temp_b;

        if (next_k_base < total_k) {
            LOAD_TILE(next_a, next_b, next_k_base);
            has_next = true;
        } else {
            has_next = false;
        }

        __syncthreads();
    }

#undef LOAD_TILE

    const uint32_t out_idx = (o * in_h + h) * in_w + w;
    output[out_idx] = __hmax(acc, 0);
}


__global__ void max_pool_nchw_cuda(const half* input,
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

    const size_t plane_size = in_h * in_w;
    const half* channel_in = input + c * plane_size;
    half* channel_out = output + c * out_h * out_w;

    half max_val = channel_in[ih * in_w + iw];
    max_val = __hmax(max_val, channel_in[ih * in_w + (iw + 1)]);
    max_val = __hmax(max_val, channel_in[(ih + 1) * in_w + iw]);
    max_val = __hmax(max_val, channel_in[(ih + 1) * in_w + (iw + 1)]);
    channel_out[h * out_w + w] = max_val;
}

__global__ void avg_pool_nchw_cuda(const half* input,
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

    const size_t plane_size = in_h * in_w;
    const half* channel_in = input + c * plane_size;
    half* channel_out = output + c * out_h * out_w;

    half avg_val = channel_in[ih * in_w + iw];
    avg_val = __hadd(avg_val, channel_in[ih * in_w + (iw + 1)]);
    avg_val = __hadd(avg_val, channel_in[(ih + 1) * in_w + iw]);
    avg_val = __hadd(avg_val, channel_in[(ih + 1) * in_w + (iw + 1)]);
    channel_out[h * out_w + w] = __hdiv(avg_val, __float2half(4.0f));
}

__global__ void nn_upsample_nchw_cuda(const half* input,
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

    const size_t in_plane = in_h * in_w;
    const size_t out_plane = out_h * out_w;
    const half* channel_in = input + c * in_plane;
    half* channel_out = output + c * out_plane;
    channel_out[h * out_w + w] = channel_in[(h / 2) * in_w + (w / 2)];
}
