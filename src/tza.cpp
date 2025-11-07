#include "tza.h"

#include <cstring>
#include <fstream>
#include <iterator>
#include <iterator>
#include <stdexcept>

constexpr uint16_t kMagic = 0x41D7;

template<typename T>
T readValue(const std::vector<uint8_t>& buffer, std::size_t& offset) {
    T value;
    std::memcpy(&value, buffer.data() + offset, sizeof(T));
    offset += sizeof(T);
    return value;
}

std::string readString(const std::vector<uint8_t>& buffer, std::size_t& offset, std::size_t count) {
    std::string str(reinterpret_cast<const char*>(buffer.data() + offset), count);
    offset += count;
    return str;
}

std::size_t elementSize(TzaTensor::DataType type) {
    switch (type) {
        case TzaTensor::DataType::Float16: return 2;
        case TzaTensor::DataType::Float32: return 4;
        default: throw std::runtime_error("Unknown tensor type");
    }
}

std::size_t TzaTensor::elementCount() const {
    std::size_t total = 1;
    for (uint32_t dim : dims)   
        total *= dim;
    return total;
}

std::size_t TzaTensor::elementSizeBytes() const {
    return elementSize(type);
}

const TzaTensor* TzaFile::find(std::string_view tensor_name) const {
    for (const auto& tensor : tensors)
        if (tensor.name == tensor_name)
            return &tensor;
    return nullptr;
}

void loadTza(const std::string& filename, TzaFile& out_file) {
    std::ifstream file(filename, std::ios::binary);
    if (!file)
        throw std::runtime_error("Cannot open TZA file: " + filename);

    std::vector<uint8_t> buffer((std::istreambuf_iterator<char>(file)),
                                std::istreambuf_iterator<char>());
    std::size_t offset = 0;

    const auto magic = readValue<uint16_t>(buffer, offset);
    if (magic != kMagic)
        throw std::runtime_error("Bad magic");

    const auto major = readValue<uint8_t>(buffer, offset);
    const auto minor = readValue<uint8_t>(buffer, offset);
    (void)major;
    (void)minor;

    const auto table_offset = readValue<uint64_t>(buffer, offset);
    offset = static_cast<std::size_t>(table_offset);

    const auto tensor_count = readValue<uint32_t>(buffer, offset);
    out_file.tensors.clear();
    out_file.tensors.reserve(tensor_count);

    for (uint32_t i = 0; i < tensor_count; ++i) {
        TzaTensor tensor;

        const auto name_len = readValue<uint16_t>(buffer, offset);
        tensor.name = readString(buffer, offset, name_len);

        const auto ndims = readValue<uint8_t>(buffer, offset);
        tensor.dims.resize(ndims);
        for (uint8_t d = 0; d < ndims; ++d)
            tensor.dims[d] = readValue<uint32_t>(buffer, offset);

        tensor.layout = readString(buffer, offset, ndims);

        const char type_char = readValue<char>(buffer, offset);
        tensor.type = (type_char == 'f') ? TzaTensor::DataType::Float32 :
                        (type_char == 'h') ? TzaTensor::DataType::Float16 :
                                            TzaTensor::DataType::Unknown;

        const auto data_offset = readValue<uint64_t>(buffer, offset);
        const auto total_bytes = tensor.elementCount() * tensor.elementSizeBytes();

        tensor.data.assign(buffer.begin() + static_cast<long>(data_offset),
                            buffer.begin() + static_cast<long>(data_offset + total_bytes));
        out_file.tensors.emplace_back(std::move(tensor));
    }
}
