#ifndef EXR_H
#define EXR_H

#include <OpenEXR/ImfArray.h>

#include <memory>
#include <string>
#include <vector>

namespace EXR { 
class Image {
public:
    void load(const std::string& filename, const std::vector<std::string>& requested_labels);

    int width = 0;
    int height = 0;

    std::vector<std::string> available_channels;
    std::vector<std::string> loaded_channels;

    std::vector<float> tensor;

private:
    std::vector<std::unique_ptr<Imf::Array2D<float>>> channel_planes;
};

void dump_image(const float* data, int width, int height, const std::string& filename, 
                const std::vector<std::string>& channels = {"R", "G", "B"});

}

#endif  // EXR_H
