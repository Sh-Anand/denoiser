#include <stdio.h>
#include "exr.h"

int main(int argc, char** argv) {
    if (argc != 2) {
        printf("Usage: %s <input_file>\n", argv[0]);
        return 1;
    }

    const char* input_file = argv[1];

    EXR exr;
    // For now, assume the beauty buffer is stored in RGB channels.
    exr.load(input_file, {"R", "G", "B", "Z", "NX", "NY", "NZ"});

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
    
    return 0;
}
