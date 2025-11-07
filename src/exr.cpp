#include "exr.h"

#include <OpenEXR/ImfChannelList.h>
#include <OpenEXR/ImfFrameBuffer.h>
#include <OpenEXR/ImfInputFile.h>
#include <Imath/ImathBox.h>

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

std::vector<std::string> gather_channel_names(const Imf::ChannelList& channels) {
    std::vector<std::string> names;
    names.reserve(8);
    for (Imf::ChannelList::ConstIterator it = channels.begin(); it != channels.end(); ++it) {
        names.emplace_back(it.name());
    }
    return names;
}

char* base_pointer(float* data, const Imath::Box2i& window, int width) {
    const std::ptrdiff_t row_stride = static_cast<std::ptrdiff_t>(width);
    return reinterpret_cast<char*>(data - static_cast<std::ptrdiff_t>(window.min.x) -
                                    static_cast<std::ptrdiff_t>(window.min.y) * row_stride);
}

}  // namespace

void EXR::load(const std::string& filename, const std::vector<std::string>& requested_labels) {
    if (requested_labels.empty()) {
        throw std::runtime_error("No channel labels requested for EXR load");
    }

    Imf::InputFile file(filename.c_str());
    const Imath::Box2i& dw = file.header().dataWindow();

    width = dw.max.x - dw.min.x + 1;
    height = dw.max.y - dw.min.y + 1;
    if (width <= 0 || height <= 0) {
        throw std::runtime_error("Invalid EXR image dimensions in: " + filename);
    }

    available_channels = gather_channel_names(file.header().channels());
    if (available_channels.empty()) {
        throw std::runtime_error("EXR file contains no channels: " + filename);
    }

    // Map channel names for quick lookup.
    std::unordered_map<std::string, int> channel_lookup;
    channel_lookup.reserve(available_channels.size());
    for (std::size_t i = 0; i < available_channels.size(); ++i) {
        channel_lookup.emplace(available_channels[i], static_cast<int>(i));
    }

    // Prepare storage for only the requested labels.
    loaded_channels.clear();
    loaded_channels.reserve(requested_labels.size());
    channel_planes.clear();
    channel_planes.reserve(requested_labels.size());

    Imf::FrameBuffer frame_buffer;
    for (const std::string& label : requested_labels) {
        auto it = channel_lookup.find(label);
        if (it == channel_lookup.end()) {
            throw std::runtime_error("Missing requested EXR channel '" + label + "' in file: " + filename);
        }

        loaded_channels.push_back(label);
        channel_planes.emplace_back(std::make_unique<Imf::Array2D<float>>());
        Imf::Array2D<float>& plane = *channel_planes.back();
        plane.resizeErase(height, width);

        frame_buffer.insert(label.c_str(),
                            Imf::Slice(Imf::FLOAT,
                                       base_pointer(&plane[0][0], dw, width),
                                       sizeof(float),
                                       sizeof(float) * static_cast<std::size_t>(width)));
    }

    file.setFrameBuffer(frame_buffer);
    file.readPixels(dw.min.y, dw.max.y);

    // Pack the requested channels into an interleaved tensor for downstream processing.
    const std::size_t channel_count = loaded_channels.size();
    tensor.assign(static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * channel_count, 0.f);
    for (std::size_t c = 0; c < channel_count; ++c) {
        Imf::Array2D<float>& plane = *channel_planes[c];
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                const std::size_t pixel_index = static_cast<std::size_t>(y) * static_cast<std::size_t>(width) + static_cast<std::size_t>(x);
                tensor[pixel_index * channel_count + c] = plane[y][x];
            }
        }
    }

    // Release the per-channel planes now that we've packed the data.
    channel_planes.clear();
}
