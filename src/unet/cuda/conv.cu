#include "conv.h"

#include <cuda_fp16.h>

__global__ void conv_relu_nhwc_oihw_cuda(const float* input,
                                     float* output,
                                     size_t in_h,
                                     size_t in_w,
                                     size_t filter_h,
                                     size_t filter_w,
                                     size_t in_c,
                                     size_t out_c,
                                     const half* weights,
                                     const half* bias) {
    const int h = blockIdx.x * blockDim.x + threadIdx.x;
    const int w = blockIdx.y * blockDim.y + threadIdx.y;
    const int o = blockIdx.z * blockDim.z + threadIdx.z;

    if (h >= in_h || w >= in_w || o >= out_c)
        return;

    const size_t pad_h = filter_h / 2;
    const size_t pad_w = filter_w / 2;

    float sum = (float) bias[o];
    for (int i = 0; i < in_c; i++) {
        for (int fh = 0; fh < filter_h; fh++) {
            const int ih = h + fh - pad_h;
            if (ih < 0 || ih >= in_h)
                continue;
            for (int fw = 0; fw < filter_w; fw++) {
                const int iw = w + fw - pad_w;
                if (iw < 0 || iw >= in_w)
                    continue;
                sum += input[(ih * in_w + iw) * in_c + i] *
                       (float) weights[o * in_c * filter_h * filter_w +
                               i * filter_h * filter_w +
                               fh * filter_w + fw];
            }
        }
    }
    output[(h * in_w + w) * out_c + o] = fmaxf(0.f, sum);
}

__global__ void max_pool_nhwc_cuda(const float* input,
                               float* output,
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

    float max_val = input[(ih * in_w + iw) * in_c + c];
    max_val = fmaxf(max_val, input[(ih * in_w + (iw + 1)) * in_c + c]);
    max_val = fmaxf(max_val, input[((ih + 1) * in_w + iw) * in_c + c]);
    max_val = fmaxf(max_val, input[((ih + 1) * in_w + (iw + 1)) * in_c + c]);
    output[(h * out_w + w) * in_c + c] = max_val;
}

__global__ void avg_pool_nhwc_cuda(const float* input,
                               float* output,
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

    float avg_val = input[(ih * in_w + iw) * in_c + c];
    avg_val += input[(ih * in_w + (iw + 1)) * in_c + c];
    avg_val += input[((ih + 1) * in_w + iw) * in_c + c];
    avg_val += input[((ih + 1) * in_w + (iw + 1)) * in_c + c];
    output[(h * out_w + w) * in_c + c] = avg_val * 0.25f;
}

__global__ void nn_upsample_nhwc_cuda(const float* input,
                                  float* output,
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

    output[(h * out_w + w) * in_c + c] = input[(h / 2 * in_w + w / 2) * in_c + c];
}
