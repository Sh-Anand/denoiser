#include <cstddef>

void conv_relu_nhwc_oihw(const float* input,
                        float* output,
                        std::size_t in_h,
                        std::size_t in_w,
                        std::size_t filter_h,
                        std::size_t filter_w,
                        std::size_t in_c,
                        std::size_t out_c,
                        const float* weights,
                        const float* bias);

void max_pool_nhwc(const float* input,
                   float* output,
                   std::size_t in_h,
                   std::size_t in_w,
                   std::size_t in_c);

void avg_pool_nhwc(const float* input,
                   float* output,
                   std::size_t in_h,
                   std::size_t in_w,
                   std::size_t in_c);

void nn_upsample_nhwc(const float* input,
                      float* output,
                      std::size_t in_h,
                      std::size_t in_w,
                      std::size_t in_c);