#ifndef EXR_H
#define EXR_H

#include <OpenEXR/ImfArray.h>

#include <memory>
#include <string>
#include <vector>

class EXR {
public:
    void load(const std::string& filename);

    int width = 0;
    int height = 0;
    int num_channels = 0;
    std::vector<std::string> channel_names;
    std::vector<std::unique_ptr<Imf::Array2D<float>>> channel_planes;
};

#endif  // EXR_H
