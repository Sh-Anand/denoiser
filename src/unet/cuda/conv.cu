#include "conv.h"

#include <cuda_fp16.h>
#include <cstdint>

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
    const bool valid = (h < in_h) &&
                       (w < in_w) &&
                       (o < out_c);

    const uint32_t total_k = in_c * 9;
    const uint32_t tile_a_elems = blockDim.x * blockDim.y * CONV_IM2COL_TILE_K;
    const uint32_t tile_b_elems = CONV_IM2COL_TILE_K * blockDim.z;
    const uint32_t max_tile = max(tile_b_elems, tile_a_elems);

    half* tile_a = smem;
    half* tile_b = tile_a + tile_a_elems;

    const half zero = __float2half(0.0f);
    half acc = valid ? bias[o] : zero;

    for (uint32_t k_base = 0; k_base < total_k; k_base += CONV_IM2COL_TILE_K) {
        for (uint32_t idx = linear_tid; idx < max_tile; idx += block_threads) {
            const uint32_t k_col = idx & (CONV_IM2COL_TILE_K - 1);
            const uint32_t k_idx = k_base + k_col;

            half val_a = zero;
            half val_b = zero;
            if (k_idx < total_k) {
                const uint32_t hw_idx = idx >> LOG_CONV_IM2COL_TILE_K;
                const uint32_t o_out = block_o0 + hw_idx;

                const uint32_t local_x = hw_idx % blockDim.x;
                const uint32_t local_y = hw_idx / blockDim.x;
                const uint32_t h_out = block_h0 + local_x;
                const uint32_t w_out = block_w0 + local_y;

                if (h_out < in_h && w_out < in_w) {
                    const uint32_t ic = k_idx / 9;
                    const uint32_t fh = (k_idx % 9) / 3;
                    const uint32_t fw = k_idx % 3;

                    const uint32_t ih = h_out + fh - 1;
                    const uint32_t iw = w_out + fw - 1;
                    if (ih < in_h && iw < in_w) {
                        const uint32_t input_idx = (ic * in_h + ih) * in_w + iw;
                        val_a = input[input_idx];
                    }
                }

                if (o_out < out_c) {
                    val_b = weights[o_out * total_k + k_idx];
                }
            }

            tile_a[idx] = val_a;
            tile_b[idx] = val_b;
        }

        __syncthreads();

        if (valid) {
            const half* a_row = tile_a + local_hw * CONV_IM2COL_TILE_K;
            const half* b_col = tile_b + threadIdx.z * CONV_IM2COL_TILE_K;
            for (uint32_t kk = 0; kk < CONV_IM2COL_TILE_K; ++kk) {
                acc += a_row[kk] * b_col[kk];
            }
        }

        __syncthreads();
    }

    if (valid) {
        const uint32_t out_idx = (o * in_h + h) * in_w + w;
        output[out_idx] = __hmax(acc, zero);
    }
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
