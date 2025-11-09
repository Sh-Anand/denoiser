#include <cstddef>
#include <omp.h>
#include <cmath>

// n = 1
void conv_relu_nhwc_oihw(const float* input,
                        float* output,
                        std::size_t in_h,
                        std::size_t in_w,
                        std::size_t filter_h,
                        std::size_t filter_w,
                        std::size_t in_c,
                        std::size_t out_c,
                        const float* weights,
                        const float* bias) {
    const std::size_t out_h = in_h - filter_h + 1;
    const std::size_t out_w = in_w - filter_w + 1;

    #pragma omp parallel for collapse(3)
    for (int h = 0; h < out_h; h++) {
        for (int w = 0; w < out_w; w++) {
            for (int o = 0; o < out_c; o++) {
                float sum = bias[o];
                for (int i = 0; i < in_c; i++) 
                    for (int fh = 0; fh < filter_h; fh++) 
                        for (int fw = 0; fw < filter_w; fw++)
                            sum += input[(h + fh) * in_w * in_c + (w + fw) * in_c + i] * weights[o * in_c * filter_h * filter_w + i * filter_h * filter_w + fh * filter_w + fw];
                output[h * out_w * out_c + w * out_c + o] = std::max(0.f, sum);
            }
        }
    }
}
