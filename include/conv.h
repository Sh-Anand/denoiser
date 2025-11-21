#include <cstddef>
#include <cstdint>

#include "cuda_defs.h"

// Centralized list of supported compile-time input channel specializations.
#ifndef FOR_EACH_IN_C
#define FOR_EACH_IN_C(M) \
    M(3) M(16) M(32) M(48) M(64) M(67) M(80) M(96) M(112) M(128) M(160)
#endif

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
                             bool cutlass_conv = false);

template <uint8_t CONV_IM2COL_TILE_K, uint8_t LOG_CONV_IM2COL_TILE_K, uint32_t IN_C>
__global__ void conv_relu_nhwc_oihw_cuda(const half* input,
                        half* output,
                        size_t in_h,
                        size_t in_w,
                        size_t out_c,
                        const half* weights,
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
