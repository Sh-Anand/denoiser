#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

#include "conv.h"

#include <omp.h>

#if defined(__SSE2__) || defined(__AVX2__) || defined(__AVX512F__)
#include <immintrin.h>
#endif

namespace {

#if defined(__AVX512F__)
struct SimdOps {
    using Vec = __m512;
    static constexpr int width = 16;

    static inline Vec load(const float* ptr) { return _mm512_loadu_ps(ptr); }
    static inline Vec mul_add(Vec acc, const float* weights, float scalar) {
        const __m512 w = _mm512_loadu_ps(weights);
        const __m512 s = _mm512_set1_ps(scalar);
        return _mm512_fmadd_ps(w, s, acc);
    }
    static inline Vec relu(Vec v) { return _mm512_max_ps(v, _mm512_setzero_ps()); }
    static inline void store(float* ptr, Vec v) { _mm512_storeu_ps(ptr, v); }
};
#elif defined(__AVX2__)
struct SimdOps {
    using Vec = __m256;
    static constexpr int width = 8;

    static inline Vec load(const float* ptr) { return _mm256_loadu_ps(ptr); }
    static inline Vec mul_add(Vec acc, const float* weights, float scalar) {
        const __m256 w = _mm256_loadu_ps(weights);
        const __m256 s = _mm256_set1_ps(scalar);
        return _mm256_add_ps(acc, _mm256_mul_ps(w, s));
    }
    static inline Vec relu(Vec v) { return _mm256_max_ps(v, _mm256_setzero_ps()); }
    static inline void store(float* ptr, Vec v) { _mm256_storeu_ps(ptr, v); }
};
#elif defined(__SSE2__)
struct SimdOps {
    using Vec = __m128;
    static constexpr int width = 4;

    static inline Vec load(const float* ptr) { return _mm_loadu_ps(ptr); }
    static inline Vec mul_add(Vec acc, const float* weights, float scalar) {
        const __m128 w = _mm_loadu_ps(weights);
        const __m128 s = _mm_set1_ps(scalar);
        return _mm_add_ps(acc, _mm_mul_ps(w, s));
    }
    static inline Vec relu(Vec v) { return _mm_max_ps(v, _mm_setzero_ps()); }
    static inline void store(float* ptr, Vec v) { _mm_storeu_ps(ptr, v); }
};
#else
struct SimdOps {
    using Vec = float;
    static constexpr int width = 1;

    static inline Vec load(const float* ptr) { return *ptr; }
    static inline Vec mul_add(Vec acc, const float* weights, float scalar) {
        return acc + weights[0] * scalar;
    }
    static inline Vec relu(Vec v) { return std::max(0.0f, v); }
    static inline void store(float* ptr, Vec v) { *ptr = v; }
};
#endif

constexpr int TILE_H = 8;
constexpr int TILE_W = 32;

} // namespace

// n = 1
void conv_relu_nhwc_oihw(const float* input,
                        float* output,
                        size_t H,
                        size_t W,
                        size_t in_c,
                        size_t out_c,
                        const _Float16* weights,
                        const _Float16* bias) {
    constexpr size_t filter_h = 3;
    constexpr size_t filter_w = 3;
    constexpr size_t pad_h = 1;
    constexpr size_t pad_w = 1;
    constexpr size_t filter_area = filter_h * filter_w;
    const size_t oc = out_c;
    const size_t ic = in_c;
    const size_t packed_size = ic * filter_area * oc;
    const size_t weights_per_ic = filter_area * oc;

    std::vector<float> bias_f32(oc);
    #pragma omp parallel for
    for (int o = 0; o < static_cast<int>(oc); ++o) {
        bias_f32[o] = static_cast<float>(bias[o]);
    }

    std::vector<float> packed_weights(packed_size);
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < static_cast<int>(ic); ++i) {
        for (int f = 0; f < static_cast<int>(filter_area); ++f) {
            const size_t dst_base = (static_cast<size_t>(i) * filter_area + static_cast<size_t>(f)) * oc;
            for (size_t o = 0; o < oc; ++o) {
                const size_t src_index = ((o * ic + static_cast<size_t>(i)) * filter_area) + static_cast<size_t>(f);
                packed_weights[dst_base + o] = static_cast<float>(weights[src_index]);
            }
        }
    }

    const int H_int = static_cast<int>(H);
    const int W_int = static_cast<int>(W);
    const float* weight_data = packed_weights.data();

    #pragma omp parallel for collapse(2)
    for (int h_tile = 0; h_tile < H_int; h_tile += TILE_H) {
        for (int w_tile = 0; w_tile < W_int; w_tile += TILE_W) {
            const int h_max = std::min(h_tile + TILE_H, H_int);
            const int w_max = std::min(w_tile + TILE_W, W_int);
            for (int h = h_tile; h < h_max; ++h) {
                for (int w = w_tile; w < w_max; ++w) {
                    float* out_pixel = output + (static_cast<size_t>(h) * W + static_cast<size_t>(w)) * oc;
                    size_t out_offset = 0;
                    if (SimdOps::width > 1) {
                        for (; out_offset + SimdOps::width <= oc; out_offset += SimdOps::width) {
                            auto sum = SimdOps::load(bias_f32.data() + out_offset);
                            for (size_t fh = 0; fh < filter_h; ++fh) {
                                const int ih = h + static_cast<int>(fh) - static_cast<int>(pad_h);
                                if (ih < 0 || ih >= H_int)
                                    continue;
                                const size_t filter_row_offset = fh * filter_w;
                                for (size_t fw = 0; fw < filter_w; ++fw) {
                                    const int iw = w + static_cast<int>(fw) - static_cast<int>(pad_w);
                                    if (iw < 0 || iw >= W_int)
                                        continue;
                                    const size_t filter_idx = filter_row_offset + fw;
                                    const size_t weight_filter_offset = filter_idx * oc;
                                    const float* in_ptr = input + (static_cast<size_t>(ih) * W + static_cast<size_t>(iw)) * ic;
                                    for (size_t c = 0; c < ic; ++c) {
                                        const float val = in_ptr[c];
                                        const float* weight_ptr = weight_data + c * weights_per_ic + weight_filter_offset + out_offset;
                                        sum = SimdOps::mul_add(sum, weight_ptr, val);
                                    }
                                }
                            }
                            sum = SimdOps::relu(sum);
                            SimdOps::store(out_pixel + out_offset, sum);
                        }
                    }
                    for (; out_offset < oc; ++out_offset) {
                        float sum = bias_f32[out_offset];
                        for (size_t fh = 0; fh < filter_h; ++fh) {
                            const int ih = h + static_cast<int>(fh) - static_cast<int>(pad_h);
                            if (ih < 0 || ih >= H_int)
                                continue;
                            const size_t filter_row_offset = fh * filter_w;
                            for (size_t fw = 0; fw < filter_w; ++fw) {
                                const int iw = w + static_cast<int>(fw) - static_cast<int>(pad_w);
                                if (iw < 0 || iw >= W_int)
                                    continue;
                                const size_t filter_idx = filter_row_offset + fw;
                                const float* in_ptr = input + (static_cast<size_t>(ih) * W + static_cast<size_t>(iw)) * ic;
                                for (size_t c = 0; c < ic; ++c) {
                                    const float* weight_ptr = weight_data + c * weights_per_ic + filter_idx * oc;
                                    sum += in_ptr[c] * weight_ptr[out_offset];
                                }
                            }
                        }
                        out_pixel[out_offset] = std::max(0.0f, sum);
                    }
                }
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
