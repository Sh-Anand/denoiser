#ifndef EXR_H
#define EXR_H

#include <OpenEXR/ImfArray.h>

#include <memory>
#include <string>
#include <vector>

namespace EXR { 
class Image {
public:
    // Loads only the requested channel labels (e.g., {"R","G","B"} or {"beauty.R", ...})
    void load(const std::string& filename, const std::vector<std::string>& requested_labels);

    int width = 0;
    int height = 0;

    // Channel metadata from the file
    std::vector<std::string> available_channels;

    // Channels actually loaded (same order as the provided labels)
    std::vector<std::string> loaded_channels;

    // Interleaved tensor storing the requested channels:
    // pixel (y,x) starts at index (y*width + x) * loaded_channels.size()
    std::vector<float> tensor;

private:
    std::vector<std::unique_ptr<Imf::Array2D<float>>> channel_planes;
};

}

#endif  // EXR_H
