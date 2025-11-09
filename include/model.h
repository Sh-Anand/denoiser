#ifndef MODEL_H
#define MODEL_H

#include "layer.h"

class UNetModel {
public:
    UNetModel(const std::string& model_name, TzaFile& weights);
    std::vector<Layer> encoder_layers;
    std::vector<Layer> decoder_layers;
    std::string model_name;
};

#endif // MODEL_H