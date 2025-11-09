#ifndef TRANSFER_H
#define TRANSFER_H

#include <cmath>

namespace Transfer {

// PU (Perceptually Uniform) transfer function from OIDN
// Used for HDR image processing
struct PU {
    static constexpr float a  =  1.41283765e+03f;
    static constexpr float b  =  1.64593172e+00f;
    static constexpr float c  =  4.31384981e-01f;
    static constexpr float d  = -2.94139609e-03f;
    static constexpr float e  =  1.92653254e-01f;
    static constexpr float f  =  6.26026094e-03f;
    static constexpr float g  =  9.98620152e-01f;
    static constexpr float y0 =  1.57945760e-06f;
    static constexpr float y1 =  3.22087631e-02f;
    static constexpr float x0 =  2.23151711e-03f;
    static constexpr float x1 =  3.70974749e-01f;

    static inline float forward(float y) {
        if (y <= y0)
            return a * y;
        else if (y <= y1)
            return std::pow(y, c) * b + d;
        else
            return std::log(y + f) * e + g;
    }

    static inline float inverse(float x) {
        if (x <= x0)
            return x / a;
        else if (x <= x1)
            return std::pow((x - d) / b, 1.0f / c);
        else
            return std::exp((x - g) / e) - f;
    }
};

// Compute normalization scale for HDR
// yMax = 65504.0f (maximum HDR value)
inline float compute_norm_scale() {
    constexpr float yMax = 65504.0f;
    float xMax = PU::forward(yMax);
    return 1.0f / xMax;
}

inline float compute_rcp_norm_scale() {
    constexpr float yMax = 65504.0f;
    return PU::forward(yMax);
}

} // namespace Transfer

#endif // TRANSFER_H
