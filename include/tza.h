#ifndef TZA_H
#define TZA_H

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

struct TzaTensorStripped {
    const uint8_t* data;
    uint32_t out_channels;
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

    const TzaTensorStripped strip(bool cuda = false) const;
};

struct TzaFile {
    std::vector<TzaTensor> tensors;

    const TzaTensor* find(std::string_view tensor_name) const;
};

void loadTza(const std::string& filename, TzaFile& out_file);

#endif // TZA_H
