#include <cstddef>
#include <omp.h>
#include <cmath>

// n = 1
void conv_relu_nhwc_oihw(const float* input,
                        float* output,
                        size_t in_h,
                        size_t in_w,
                        size_t filter_h,
                        size_t filter_w,
                        size_t in_c,
                        size_t out_c,
                        const float* weights,
                        const float* bias) {
    const size_t out_h = in_h - filter_h + 1;
    const size_t out_w = in_w - filter_w + 1;

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

// 2x2 max pool stride 2
// in_h and in_w must be even
void max_pool_nhwc(const float* input,
                   float* output,
                   size_t in_h,
                   size_t in_w,
                   size_t in_c) {
    const size_t out_h = in_h / 2;
    const size_t out_w = in_w / 2;

    #pragma omp parallel for collapse(3)
    for (int h = 0; h < out_h; h++) {
        const size_t h0 = 2 * h;
        for (int w = 0; w < out_w; w++) {
            const size_t w0 = 2 * w;
            for (int c = 0; c < in_c; c++) {
                float max_val = input[(h0 * in_w + w0) * in_c + c];
                max_val = std::max(max_val, input[(h0 * in_w + (w0 + 1)) * in_c + c]);
                max_val = std::max(max_val, input[((h0 + 1) * in_w + w0) * in_c + c]);
                max_val = std::max(max_val, input[((h0 + 1) * in_w + (w0 + 1)) * in_c + c]);
                output[(h * out_w + w) * in_c + c] = max_val;
            }
        }
    }
}

// 2x2 avg pool stride 2
// in_h and in_w must be even
void avg_pool_nhwc(const float* input,
                   float* output,
                   size_t in_h,
                   size_t in_w,
                   size_t in_c) {
    const size_t out_h = in_h / 2;
    const size_t out_w = in_w / 2;

    #pragma omp parallel for collapse(3)
    for (int h = 0; h < out_h; h++) {
        const size_t h0 = 2 * h;
        for (int w = 0; w < out_w; w++) {
            const size_t w0 = 2 * w;
            for (int c = 0; c < in_c; c++) {
                float avg_val = input[(h0 * in_w + w0) * in_c + c];
                avg_val += input[(h0 * in_w + (w0 + 1)) * in_c + c];
                avg_val += input[((h0 + 1) * in_w + w0) * in_c + c];
                avg_val += input[((h0 + 1) * in_w + (w0 + 1)) * in_c + c];
                output[(h * out_w + w) * in_c + c] = avg_val * 0.25f;
            }
        }
    }
}

// 2x2 NN upsample stride 2
// in_h and in_w must be even
void nn_upsample_nhwc(const float* input,
                      float* output,
                      size_t in_h,
                      size_t in_w,
                      size_t in_c) {
    const size_t out_h = in_h * 2;
    const size_t out_w = in_w * 2;

    #pragma omp parallel for collapse(3)
    for (int h = 0; h < out_h; h++) {
        const size_t h0 = h / 2;
        for (int w = 0; w < out_w; w++) {
            const size_t w0 = w / 2;
            for (int c = 0; c < in_c; c++) {
                output[(h * out_w + w) * in_c + c] = input[(h0 * in_w + w0) * in_c + c];
            }
        }
    }
}
