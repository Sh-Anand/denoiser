#ifndef MODEL_H
#define MODEL_H

#include "layer.h"
#include <vector>

struct UNetModel {
    // if cuda, all pointers in CUDA mem
    std::vector<Layer> encoder_layers;
    std::vector<Layer> decoder_layers;

    TensorData* weights;
};

UNetModel createUNetModel(const std::string& model_name, TzaFile& weights, bool cuda = false);
void freeUNetModel(UNetModel& model, bool cuda = false);

#endif // MODEL_H