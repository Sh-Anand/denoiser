#include <cstddef>

#include "cuda_defs.h"

void conv_relu_nhwc_oihw(const float* input,
                        float* output,
                        size_t in_h,
                        size_t in_w,
                        size_t filter_h,
                        size_t filter_w,
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


DEVICE void conv_relu_nhwc_oihw_cuda(const float* input,
                        float* output,
                        size_t in_h,
                        size_t in_w,
                        size_t filter_h,
                        size_t filter_w,
                        size_t in_c,
                        size_t out_c,
                        const _Float16* weights,
                        const _Float16* bias);

DEVICE void max_pool_nhwc_cuda(const float* input,
                   float* output,
                   size_t in_h,
                   size_t in_w,
                   size_t in_c);

DEVICE void avg_pool_nhwc_cuda(const float* input,
                   float* output,
                   size_t in_h,
                   size_t in_w,
                   size_t in_c);

DEVICE void nn_upsample_nhwc_cuda(const float* input,
                      float* output,
                      size_t in_h,
                      size_t in_w,
                      size_t in_c);