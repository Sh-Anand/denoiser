#ifndef TZA_H
#define TZA_H

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#ifdef __CUDACC__
#include <cuda_fp16.h>
#endif

union TensorData {
    #ifdef __CUDACC__
    const half* half_data;
    #endif
    const _Float16* float16_data;
};

struct TzaTensor {
    enum class DataType
    {
        Float16,
        Float32,
        Unknown
    };

    std::string name;
    std::vector<uint32_t> dims;
    std::string layout;
    DataType type = DataType::Unknown;
    std::vector<uint8_t> data; // raw bytes, tightly packed

    std::size_t elementCount() const;
    std::size_t elementSizeBytes() const;

};

struct TzaFile {
    std::vector<TzaTensor> tensors;

    const TzaTensor* find(std::string_view tensor_name) const;
};

void loadTza(const std::string& filename, TzaFile& out_file);

#endif // TZA_H
