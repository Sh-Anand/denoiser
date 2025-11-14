#include "conv.h"

#include <cuda_fp16.h>

__global__ void conv_relu_nhwc_oihw_cuda(const half* input,
                                         half* output,
                                         size_t in_h,
                                         size_t in_w,
                                         size_t in_c,
                                         size_t out_c,
                                         const half* weights,
                                         const half* bias) {
    extern __shared__ half smem[];

    const int block_threads = blockDim.x * blockDim.y * blockDim.z;
    const int linear_tid = (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
    const int local_hw = threadIdx.y * blockDim.x + threadIdx.x;

    const int block_h0 = blockIdx.x * blockDim.x;
    const int block_w0 = blockIdx.y * blockDim.y;
    const int block_o0 = blockIdx.z * blockDim.z;

    const int h = block_h0 + threadIdx.x;
    const int w = block_w0 + threadIdx.y;
    const int o = block_o0 + threadIdx.z;
    const bool valid = (h < in_h) &&
                       (w < in_w) &&
                       (o < out_c);

    const size_t total_k = in_c * 9;
    const int tile_a_elems = blockDim.x * blockDim.y * CONV_IM2COL_TILE_K;
    const int tile_b_elems = CONV_IM2COL_TILE_K * blockDim.z;

    half* tile_a = smem;
    half* tile_b = tile_a + tile_a_elems;

    const half zero = __float2half(0.0f);
    half acc = valid ? bias[o] : zero;

    for (size_t k_base = 0; k_base < total_k; k_base += CONV_IM2COL_TILE_K) {
        for (int idx = linear_tid; idx < tile_a_elems; idx += block_threads) {
            const int hw_idx = idx / CONV_IM2COL_TILE_K;
            const int k_col = idx % CONV_IM2COL_TILE_K;
            const size_t k_idx = k_base + k_col;


            // guarantee required that TILE_K is divisible by block.Z
            half val = zero;
            const int local_x = hw_idx % blockDim.x;
            const int local_y = hw_idx / blockDim.x;
            const int h_out = block_h0 + local_x;
            const int w_out = block_w0 + local_y;

            if (h_out < in_h && w_out < in_w) {
                const int ic = k_idx / 9;
                const int fh = (k_idx % 9) / 3;
                const int fw = k_idx % 3;

                const int ih = h_out + fh - 1;
                const int iw = w_out + fw - 1;
                if (ih >= 0 && ih < in_h &&
                    iw >= 0 && iw < in_w) {
                    const size_t input_idx = (ih * in_w + iw) * in_c + ic;
                    val = input[input_idx];
                }
            }

            tile_a[idx] = val;
        }

        for (int idx = linear_tid; idx < tile_b_elems; idx += block_threads) {
            const int out_ch = idx / CONV_IM2COL_TILE_K;
            const int k_row = idx % CONV_IM2COL_TILE_K;
            const size_t k_idx = k_base + k_row;
            const int o_out = block_o0 + out_ch;

            half val = zero;
            if (k_idx < total_k && o_out < out_c) {
                val = weights[o_out * total_k + k_idx];
            }

            tile_b[idx] = val;
        }

        __syncthreads();

        if (valid) {
            const half* a_row = tile_a + local_hw * CONV_IM2COL_TILE_K;
            const half* b_col = tile_b + threadIdx.z * CONV_IM2COL_TILE_K;
            for (int kk = 0; kk < CONV_IM2COL_TILE_K; ++kk) {
                acc += a_row[kk] * b_col[kk];
            }
        }

        __syncthreads();
    }

    if (valid) {
        output[(h * in_w + w) * out_c + o] = __hmax(acc, zero);
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

    half max_val = input[(ih * in_w + iw) * in_c + c];
    max_val = __hmax(max_val, input[(ih * in_w + (iw + 1)) * in_c + c]);
    max_val = __hmax(max_val, input[((ih + 1) * in_w + iw) * in_c + c]);
    max_val = __hmax(max_val, input[((ih + 1) * in_w + (iw + 1)) * in_c + c]);
    output[(h * out_w + w) * in_c + c] = max_val;
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

    half avg_val = input[(ih * in_w + iw) * in_c + c];
    avg_val = __hadd(avg_val, input[(ih * in_w + (iw + 1)) * in_c + c]);
    avg_val = __hadd(avg_val, input[((ih + 1) * in_w + iw) * in_c + c]);
    avg_val = __hadd(avg_val, input[((ih + 1) * in_w + (iw + 1)) * in_c + c]);
    output[(h * out_w + w) * in_c + c] = __hdiv(avg_val, 4.0f);
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

    output[(h * out_w + w) * in_c + c] = input[((h / 2) * in_w + (w / 2)) * in_c + c];
}
