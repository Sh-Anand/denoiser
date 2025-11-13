#include <cstddef>

#include "cuda_defs.h"

constexpr int CONV_IM2COL_TILE_K = 16;

void conv_relu_nhwc_oihw(const float* input,
                        float* output,
                        size_t in_h,
                        size_t in_w,
                        size_t in_c,
                        size_t out_c,
                        const _Float16* weights,
                        const _Float16* bias);

void max_pool_nhwc(const float* input,
                   float* output,
                   size_t in_h,
                   size_t in_w,
                   size_t in_c);

void avg_pool_nhwc(const float* input,
                   float* output,
                   size_t in_h,
                   size_t in_w,
                   size_t in_c);

void nn_upsample_nhwc(const float* input,
                      float* output,
                      size_t in_h,
                      size_t in_w,
                      size_t in_c);


#ifdef __CUDACC__
#include <cuda_fp16.h>

__global__ void conv_relu_nhwc_oihw_cuda(const half* input,
                        half* output,
                        size_t in_h,
                        size_t in_w,
                        size_t in_c,
                        size_t out_c,
                        const __half* weights,
                        const __half* bias);

__global__ void max_pool_nhwc_cuda(const half* input,
                   half* output,
                   size_t in_h,
                   size_t in_w,
                   size_t in_c);

__global__ void avg_pool_nhwc_cuda(const half* input,
                   half* output,
                   size_t in_h,
                   size_t in_w,
                   size_t in_c);

__global__ void nn_upsample_nhwc_cuda(const half* input,
                      half* output,
                      size_t in_h,
                      size_t in_w,
                      size_t in_c);
#endif
