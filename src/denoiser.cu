#include <stdio.h>
#include <string>

#include "exr.h"
#include "tza.h"
#include "unet.h"

int main(int argc, char** argv) {
    setbuf(stdout, NULL);
    
    if (argc < 2 || argc > 3) {
        printf("Usage: %s <input_file> [weights.tza]\n", argv[0]);
        return 1;
    }

    const char* input_file = argv[1];
    std::string weights_path = (argc == 3) ? argv[2] : "../weights/rt_hdr.tza";

    EXR::Image exr;
    // For now, assume the beauty buffer is stored in RGB channels.
    exr.load(input_file, {"R", "G", "B"});

    printf("Width: %d\n", exr.width);
    printf("Height: %d\n", exr.height);
    printf("Requested channels: %zu\n", exr.loaded_channels.size());
    printf("Available channels: ");
    for (const auto& name : exr.available_channels) {
        printf("%s ", name.c_str());
    }
    printf("\n");

    printf("Loaded channels: ");
    for (const auto& name : exr.loaded_channels) {
        printf("%s ", name.c_str());
    }
    printf("\n");

    printf("Tensor size: %zu floats\n", exr.tensor.size());

    TzaFile weights;
    loadTza(weights_path, weights); 

    printf("Loaded %zu tensors from %s\n", weights.tensors.size(), weights_path.c_str());
    for (const auto& tensor : weights.tensors) {
        printf("  %s : type=%s dims=[", tensor.name.c_str(),
               tensor.type == TzaTensor::DataType::Float32 ? "f32" :
               tensor.type == TzaTensor::DataType::Float16 ? "f16" : "unknown");
        for (std::size_t i = 0; i < tensor.dims.size(); ++i) {
            printf("%u", tensor.dims[i]);
            if (i + 1 < tensor.dims.size())
                printf(",");
        }
        printf("]\n");
    }

    EXR::Image output_img;
    oidn_unet(exr, weights, output_img);
    
    std::string output_file = "output.exr";
    EXR::dump_image(output_img.tensor.data(), output_img.width, output_img.height, output_file);
    printf("Saved output to %s\n", output_file.c_str());
    
    return 0;
}
