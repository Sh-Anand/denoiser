#include <stdio.h>
#include <string>

#include "exr.h"
#include "model.h"
#include "tza.h"
#include "unet.h"

int main(int argc, char** argv) {
    setbuf(stdout, NULL);
    
    if (argc < 2 || argc > 5) {
        printf("Usage: %s <input_file> [target=cpu|cuda] [model_name=rt_hdr] [output_file=output.exr]\n", argv[0]);
        return 1;
    }

    const char* input_file = argv[1];
    std::string target = (argc >= 3) ? argv[2] : "cpu";
    std::string model_name = (argc >= 4) ? argv[3] : "rt_hdr";
    std::string weights_path = "../weights/" + model_name + ".tza";
    std::string output_file = (argc >= 5) ? argv[4] : "output.exr";

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

    UNetModel model = createUNetModel(model_name, weights, target == "cuda");
    float* output_img;

    if (target == "cpu") {
        oidn_unet(exr, model, output_img);
    } else if (target == "cuda") {
        oidn_unet_cuda(exr, model, output_img);
    } else {
        printf("Error: Unknown target '%s'. Use 'cpu' or 'cuda'.\n", target.c_str());
        return 1;
    }
    
    EXR::dump_image(output_img, exr.width, exr.height, output_file);
    printf("Saved output to %s\n", output_file.c_str());
    
    return 0;
}
