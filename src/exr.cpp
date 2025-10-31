#include "exr.h"

#include <OpenEXR/ImfChannelList.h>
#include <OpenEXR/ImfFrameBuffer.h>
#include <OpenEXR/ImfInputFile.h>
#include <Imath/ImathBox.h>

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>
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

void EXR::load(const std::string& filename) {
    Imf::InputFile file(filename.c_str());
    const Imath::Box2i& dw = file.header().dataWindow();

    width = dw.max.x - dw.min.x + 1;
    height = dw.max.y - dw.min.y + 1;
    if (width <= 0 || height <= 0) {
        throw std::runtime_error("Invalid EXR image dimensions in: " + filename);
    }

    channel_names = gather_channel_names(file.header().channels());
    if (channel_names.empty()) {
        throw std::runtime_error("EXR file contains no channels: " + filename);
    }

    num_channels = static_cast<int>(channel_names.size());

    channel_planes.clear();
    channel_planes.reserve(static_cast<std::size_t>(num_channels));
    for (int c = 0; c < num_channels; ++c) {
        channel_planes.emplace_back(std::make_unique<Imf::Array2D<float>>());
        channel_planes.back()->resizeErase(height, width);
    }

    Imf::FrameBuffer frame_buffer;
    for (int c = 0; c < num_channels; ++c) {
        Imf::Array2D<float>& plane = *channel_planes[static_cast<std::size_t>(c)];
        frame_buffer.insert(channel_names[static_cast<std::size_t>(c)].c_str(),
                            Imf::Slice(Imf::FLOAT,
                                       base_pointer(&plane[0][0], dw, width),
                                       sizeof(float),
                                       sizeof(float) * static_cast<std::size_t>(width)));
    }

    file.setFrameBuffer(frame_buffer);
    file.readPixels(dw.min.y, dw.max.y);
}
