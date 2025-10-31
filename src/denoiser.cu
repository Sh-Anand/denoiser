#include <stdio.h>
#include "exr.h"

int main(int argc, char** argv) {
    if (argc != 2) {
        printf("Usage: %s <input_file>\n", argv[0]);
        return 1;
    }

    const char* input_file = argv[1];

    EXR exr;
    exr.load(input_file);

    printf("Width: %d\n", exr.width);
    printf("Height: %d\n", exr.height);
    printf("Num channels: %d\n", exr.num_channels);
    printf("Channel names: ");
    for (const auto& name : exr.channel_names) {
        printf("%s ", name.c_str());
    }
    printf("\n");
    
    return 0;
}