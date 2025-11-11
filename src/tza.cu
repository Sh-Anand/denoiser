#include "tza.h"

const TzaTensorStripped TzaTensor::strip(bool cuda) const {
    if (cuda) {
        uint8_t* d_data;
        cudaMalloc(&d_data, data.size());
        cudaMemcpy(d_data, data.data(), data.size(), cudaMemcpyHostToDevice);
        return TzaTensorStripped{TzaTensorStrippedData{half_data: (const half*)(d_data)}, dims[0]};
    }
    return TzaTensorStripped{TzaTensorStrippedData{float16_data: (const _Float16*)(data.data())}, dims[0]};
}